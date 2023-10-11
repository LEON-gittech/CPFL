# -*- coding: utf-8 -*-
import copy

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

import pcode.create_dataset as create_dataset
import pcode.create_metrics as create_metrics
import pcode.create_model as create_model
import pcode.create_optimizer as create_optimizer
import pcode.create_scheduler as create_scheduler
import pcode.datasets.mixup_data as mixup
import pcode.local_training.compressor as compressor
import pcode.local_training.random_reinit as random_reinit
from pcode.utils.logging import display_training_stat
from pcode.utils.stat_tracker import RuntimeTracker
from pcode.utils.tensor_buffer import TensorBuffer
from pcode.utils.timer import Timer
from pcode.workers.worker_base import WorkerBase

class WorkerFedAvgCon(WorkerBase):
    def _train(self):
        self.model.train()
        # init the model and dataloader.
        self.prepare_local_train_loader()
        if self.conf.graph.on_cuda:
            self.model = self.model.cuda()

        # define optimizer, scheduler and runtime tracker.
        self.optimizer = create_optimizer.define_optimizer(
            self.conf, model=self.model, optimizer_name=self.conf.optimizer
        )
        self.scheduler = create_scheduler.Scheduler(self.conf, optimizer=self.optimizer)
        self.tracker = RuntimeTracker(metrics_to_track=self.metrics.metric_names)
        self.conf.logger.log(
            f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) enters the local training phase (current communication rounds={self.conf.graph.comm_round})."
        )

        # efficient local training.
        if hasattr(self, "model_compression_fn"):
            self.model_compression_fn.compress_model(
                param_groups=self.optimizer.param_groups
            )
        
        # entering local updates and will finish only after reaching the expected local_n_epochs.
        while True:
            for _input, _target in self.train_loader:
                # load data
                with self.timer("load_data", epoch=self.scheduler.epoch_):
                    data_batch = create_dataset.load_data_batch(
                        self.conf, _input, _target, is_training=True
                    )

                # inference and get current performance.
                with self.timer("forward_pass", epoch=self.scheduler.epoch_):
                    self.optimizer.zero_grad()
                    loss, output = self._inference(data_batch)

                    if self.conf.local_prox_term != 0:
                        loss += self.cal_prox_loss()

                with self.timer("backward_pass", epoch=self.scheduler.epoch_):
                    loss.backward()
                    # self._add_grad_from_prox_regularized_loss()
                    self.optimizer.step()
                    self.scheduler.step()

                # efficient local training.
                with self.timer("compress_model", epoch=self.scheduler.epoch_):
                    if hasattr(self, "model_compression_fn"):
                        self.model_compression_fn.compress_model(
                            param_groups=self.optimizer.param_groups
                        )

                # display the logging info.
                # display_training_stat(self.conf, self.scheduler, self.tracker)

                # display tracking time.
                if (
                    self.conf.display_tracked_time
                    and self.scheduler.local_index % self.conf.summary_freq == 0
                ):
                    self.conf.logger.log(self.timer.summary())

                # check divergence.
                if self.tracker.stat["loss"].avg > 1e3 or np.isnan(
                    self.tracker.stat["loss"].avg
                ):
                    self.conf.logger.log(
                        f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) diverges!!!!!Early stop it."
                    )
                    self._terminate_comm_round()
                    return

                # check stopping condition.
                if self._is_finished_one_comm_round():
                    display_training_stat(self.conf, self.scheduler, self.tracker)
                    self._terminate_comm_round()
                    return
                
            display_training_stat(self.conf, self.scheduler, self.tracker)
            
            # refresh the logging cache at the end of each epoch.
            self.tracker.reset()
            if self.conf.logger.meet_cache_limit():
                self.conf.logger.save_json()

    def cal_prox_loss(self):
        prox_term = 0.
        for w, w_t in zip(self.model.parameters(), self.init_model.parameters()):
            prox_term += torch.pow(torch.norm((w - w_t)), 2)

        return (self.conf.local_prox_term / 2) * prox_term
    
    def prepare_local_train_loader(self):
        if self.conf.prepare_data == "combine":
            self.train_loader = create_dataset.define_local_data_loader(
                self.conf,
                self.conf.graph.client_id,
                data_type = "train",
                data=self.local_datasets[self.conf.graph.client_id]["train"],
            )
        else:
            self.train_loader, _ = create_dataset.define_data_loader(
                self.conf,
                dataset=self.dataset["train"],
                # localdata_id start from 0 to the # of clients - 1.
                # client_id starts from 1 to the # of clients.
                localdata_id=self.conf.graph.client_id - 1,
                is_train=True,
                data_partitioner=self.data_partitioner,
            )

    def _inference(self, data_batch):
        """Inference on the given model and get loss and accuracy."""
        # do the forward pass and get the output.
        output = self.model(data_batch["input"])

        # evaluate the output and get the loss, performance.
        if self.conf.use_mixup:
            loss = mixup.mixup_criterion(
                self.criterion,
                output,
                data_batch["target_a"],
                data_batch["target_b"],
                data_batch["mixup_lambda"],
            )

            performance_a = self.metrics.evaluate(loss, output, data_batch["target_a"])
            performance_b = self.metrics.evaluate(loss, output, data_batch["target_b"])
            performance = [
                data_batch["mixup_lambda"] * _a + (1 - data_batch["mixup_lambda"]) * _b
                for _a, _b in zip(performance_a, performance_b)
            ]
        else:
            loss = self.criterion(output, data_batch["target"])
            performance = self.metrics.evaluate(loss, output, data_batch["target"])

        # update tracker.
        if self.tracker is not None:
            self.tracker.update_metrics(
                [loss.item()] + performance, n_samples=data_batch["input"].size(0)
            )
        return loss, output

    def _local_training_with_self_distillation(self, loss, output, data_batch):
        if self.conf.self_distillation > 0:
            loss = loss * (
                1 - self.conf.self_distillation
            ) + self.conf.self_distillation * self._divergence(
                student_logits=output / self.conf.self_distillation_temperature,
                teacher_logits=self.init_model(data_batch["input"])
                / self.conf.self_distillation_temperature,
            )
        return loss

    def _divergence(self, student_logits, teacher_logits, KL_temperature=1.0):
        divergence = F.kl_div(
            F.log_softmax(student_logits / KL_temperature, dim=1),
            F.softmax(teacher_logits / KL_temperature, dim=1),
            reduction="batchmean",
        )  # forward KL
        return KL_temperature * KL_temperature * divergence

    def _turn_off_grad(self, model):
        for param in model.parameters():
            param.requires_grad = False
        return model

    def _send_model_to_master(self):
        dist.barrier()
        self.conf.logger.log(
            f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) sending the model ({self.arch}) back to Master."
        )
        flatten_model = TensorBuffer(list(self.model.state_dict().values()))
        dist.send(tensor=flatten_model.buffer, dst=0)
        dist.barrier()
    
    def get_next_train_batch(self):
        try:
            # Samples a new batch for persionalizing
            (X, y) = next(self.iter_trainloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_trainloader = iter(self.train_loader)
            (X, y) = next(self.iter_trainloader)
        return (X.to(self.device), y.to(self.device))
    
    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def _terminate_comm_round(self):
        self.model = self.model.cpu()
        if hasattr(self, 'init_model'):
            del self.init_model
        self.scheduler.clean()
        self.conf.logger.save_json()
        self.conf.logger.log(
            f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) finished one round of federated learning: (comm_round={self.conf.graph.comm_round})."
        )

    def _terminate_by_early_stopping(self):
        if self.conf.graph.comm_round == -1:
            dist.barrier()
            self.conf.logger.log(
                f"Worker-{self.conf.graph.worker_id} finished the federated learning by early-stopping."
            )
            return True
        else:
            return False

    def _terminate_by_complete_training(self):
        if self.conf.graph.comm_round == self.conf.n_comm_rounds:
            dist.barrier()
            self.conf.logger.log(
                f"Worker-{self.conf.graph.worker_id} finished the federated learning: (total comm_rounds={self.conf.graph.comm_round})."
            )
            return True
        else:
            return False

    def _is_finished_one_comm_round(self):
        return True if self.conf.epoch_ >= self.conf.local_n_epochs else False
