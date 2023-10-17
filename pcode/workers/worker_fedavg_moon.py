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

class WorkerFedAvgMoon(WorkerBase):
    def __init__(self, conf):
        super().__init__(conf)
        self.cos = torch.nn.CosineSimilarity(dim=-1)
        
    def _listen_to_master(self):
        # listen to master, related to the function `_activate_selected_clients` in `master.py`.
        msg = torch.zeros((3, self.conf.n_participated))
        dist.broadcast(tensor=msg, src=0)
        self.conf.graph.client_id, self.conf.graph.comm_round, self.n_local_epochs = (
            msg[:, self.conf.graph.rank - 1].to(int).cpu().numpy().tolist()
        )

        # once we receive the signal, we init for the local training.
        self.arch, self.model = create_model.define_model(
            self.conf, to_consistent_model=False, client_id=self.conf.graph.client_id
        )
        _, self.global_model = create_model.define_model(
            self.conf, to_consistent_model=False, client_id=self.conf.graph.client_id
        )
        self.model_state_dict = self.model.state_dict()
        self.model_tb = TensorBuffer(list(self.model_state_dict.values()))
        self.metrics = create_metrics.Metrics(self.model, task="classification")
        dist.barrier()

    def _recv_model_from_master(self):
        # related to the function `_send_model_to_selected_clients` in `master.py`
        old_buffer = copy.deepcopy(self.model_tb.buffer)#通过计算模型参数的范数来判断模型是否更新
        dist.recv(self.model_tb.buffer, src=0)
        new_buffer = copy.deepcopy(self.model_tb.buffer)
        self.model_tb.unpack(self.model_state_dict.values())
        self.model.load_state_dict(self.model_state_dict)
        #保存不参与本地更新的 global model
        self.global_model.load_state_dict(self.model_state_dict)
        #如果是第一轮，则last_local_model 就是 master 发来的 model
        if self.conf.graph.comm_round == 1:
            self.last_local_model = copy.deepcopy(self.model)
        
        random_reinit.random_reinit_model(self.conf, self.model)
        self.init_model = self._turn_off_grad(copy.deepcopy(self.model).to(self.device))
        self.conf.logger.log(
            f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) received the global/personal model ({self.arch}) from Master. The model status {'is updated' if old_buffer.norm() != new_buffer.norm() else 'is not updated'}."
        )        
        dist.barrier() 
    
    def _send_model_to_master(self):
        dist.barrier()
        self.conf.logger.log(
            f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) sending the model ({self.arch}) back to Master."
        )
        #更新 last_local_model
        self.last_local_model.load_state_dict(self.model.state_dict())
        
        flatten_model = TensorBuffer(list(self.model.state_dict().values()))
        dist.send(tensor=flatten_model.buffer, dst=0)
        dist.barrier() 
        
    def _train(self):
        self.model.train()
        self.global_model.eval()
        self.last_local_model.eval()
        # init the model and dataloader.
        self.prepare_local_train_loader()
        if self.conf.graph.on_cuda:
            self.model = self.model.cuda()
            self.global_model=self.global_model.cuda()
            self.last_local_model=self.last_local_model.cuda()

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
    
    def _inference(self, data_batch):
        """Inference on the given model and get loss and accuracy."""
        # do the forward pass and get the output.
        # output = self.model(data_batch["pos1"],data_batch["pos2"],data_batch["input"])
        _, pro1,out  = self.model(data_batch["input"])
        _, pro2,_ = self.global_model(data_batch["input"])
        
        posi = self.cos(pro1, pro2)#[B]
        logits = posi.reshape(-1,1)#[B,1]

        _, pro3, _ = self.last_local_model(data_batch["input"])
        nega = self.cos(pro1, pro3)
        
        # moon算是一种二分类任务
        logits = torch.cat((logits, nega.reshape(-1,1)), dim=1)

        logits /= self.conf.temperature
        #改一下分子
        labels = torch.ones(data_batch["input"].size(0)).cuda().long()
        # labels = torch.zeros(data_batch["input"].size(0)).cuda().long()

        loss2 = self.conf.mu * self.criterion(logits, labels)


        loss1 = self.criterion(out, data_batch["target"])
        loss = loss1 + loss2

        performance = self.metrics.evaluate(loss, out, data_batch["target"])

        # update tracker.
        if self.tracker is not None:
            self.tracker.update_metrics(
                [loss.item()] + performance, n_samples=data_batch["input"].size(0)
            )
        return loss, out