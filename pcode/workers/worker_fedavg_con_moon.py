# -*- coding: utf-8 -*-
import copy

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.modules import loss
from tqdm import tqdm

import pcode.create_aggregator as create_aggregator
import pcode.create_dataset as create_dataset
import pcode.create_metrics as create_metrics
import pcode.create_model as create_model
import pcode.create_optimizer as create_optimizer
import pcode.create_scheduler as create_scheduler
import pcode.local_training.random_reinit as random_reinit
import pcode.models as models
from pcode import master_utils
from pcode.utils.logging import display_training_stat
from pcode.utils.stat_tracker import RuntimeTracker
from pcode.utils.tensor_buffer import TensorBuffer
from pcode.workers.worker_base import WorkerBase


class WorkerFedAvgConMoon(WorkerBase):
    def __init__(self, conf):
        super().__init__(conf)
        self.cos = torch.nn.CosineSimilarity(dim=-1)
        self.M = 1 #应该是保存多少个先前阶段的模型

    def run(self):
        while True:
            self._listen_to_master() #每一轮的初始化

            # check if we need to terminate the training or not.
            if self._terminate_by_early_stopping():
                return

            self._recv_model_from_master()

            if self.is_active_before == 0:
                self._train()
            else:
                self._train_AKT()
                
            self._send_model_to_master()

            # check if we need to terminate the training or not.
            if self._terminate_by_complete_training():
                return
    
    #重写train
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
            for pos1,pos2,_input, _target in self.train_loader:
                # load data
                with self.timer("load_data", epoch=self.scheduler.epoch_):
                    data_batch = create_dataset.load_data_batch_con(
                        self.conf, pos1,pos2,_input, _target, is_training=True
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
        feature_1, out_1,pred_c_1  = self.model(data_batch["pos1"])
        feature_2, out_2,pred_c_2 = self.model(data_batch["pos2"])
        _, pro1 ,output = self.model(data_batch["input"])
        loss = 0
        if self.is_active_before == 1:
            _, pro2, _ = self.global_model(data_batch["input"])
            posi = self.cos(pro1, pro2)#[B]
            logits = posi.reshape(-1,1)#[B,1]

            _, pro3, _ = self.last_local_model(data_batch["input"])
            nega = self.cos(pro1, pro3)
            
            # moon算是一种二分类任务
            logits = torch.cat((logits, nega.reshape(-1,1)), dim=1)

            logits /= self.conf.temperature
            #改一下分子
            labels = torch.ones(data_batch["input"].size(0)).cuda().long()

            moon_loss = self.conf.mu * self.criterion(logits, labels)
            loss += moon_loss
            # print("moon_loss:",moon_loss)
            
        # [2*B, D]
        out = torch.cat([out_1, out_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / self.conf.temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * self.conf.batch_size, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * self.conf.batch_size, -1)

        # compute loss
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.conf.temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        con_loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()* self.conf.lam
        # print("con_loss:",con_loss)
        loss += con_loss
        #loss 加上 CEL
        pred_c = (pred_c_1+pred_c_2)/2
        tmp = F.one_hot(data_batch["target"].cuda(non_blocking=True,device=self.device),num_classes=self.conf.num_classes).float()
        cel = F.cross_entropy(pred_c,tmp)
        # print("cel:",cel)
        loss += cel

        performance = self.metrics.evaluate(loss, output, data_batch["target"])

        # update tracker.
        if self.tracker is not None:
            self.tracker.update_metrics(
                [loss.item()] + performance, n_samples=data_batch["input"].size(0)
            )
        return loss, output

    
    def _listen_to_master(self): #每一轮的初始化
        # listen to master, related to the function `_activate_selected_clients` in `master.py`.
        msg = torch.zeros((4, self.conf.n_participated))
        dist.broadcast(tensor=msg, src=0)
        self.conf.graph.client_id, self.conf.graph.comm_round, self.n_local_epochs, self.is_active_before= (
            msg[:, self.conf.graph.rank - 1].to(int).cpu().numpy().tolist()
        )

        # once we receive the signal, we init for the local training.
        self.arch, self.model = create_model.define_model(
            self.conf, to_consistent_model=False, client_id=self.conf.graph.client_id #这里的 client_id 用于在使用复杂结构，即不同主机的模型结构不一致时，决定不同主机使用哪个模型，对于统一模型结构没有什么用
        )
        self.model_state_dict = self.model.state_dict()
        self.model_tb = TensorBuffer(list(self.model_state_dict.values()))
        self.metrics = create_metrics.Metrics(self.model, task="classification")
        dist.barrier()
    

    def _recv_model_from_master(self):
        # related to the function `_send_model_to_selected_clients` in `master.py`
        dist.recv(self.model_tb.buffer, src=0)
        self.model_tb.unpack(self.model_state_dict.values())
        self.model.load_state_dict(self.model_state_dict)
        self.global_model = copy.deepcopy(self.model)
        self.conf.logger.log(
            f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) received the global/personal model ({self.arch}) from Master."
        )
        
        if self.is_active_before == 1: #第一轮就不需要了
            # init the placeholders to recv the other local models from master.
            flatten_local_models = []
            for i in range(self.M):
                client_tb = TensorBuffer(
                    list(copy.deepcopy(self.model.state_dict()).values())
                )
                client_tb.buffer = torch.zeros_like(client_tb.buffer)
                flatten_local_models.append(client_tb)
            # receive local models from master. #接收上一轮的本地模型
            for i in range(self.M):
                dist.recv(tensor=flatten_local_models[i].buffer, src=0)
            
            self.last_local_model = copy.deepcopy(self.model)
            _last_model_state_dict = self.last_local_model.state_dict()
            flatten_local_models[0].unpack(_last_model_state_dict.values())
            self.last_local_model.load_state_dict(_last_model_state_dict)

            self.conf.logger.log(
                f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) received ({self.M}) past local models from Master."
            )
      
        dist.barrier()  

    def _train_AKT(self):
        self.model.train()

        # init the model and dataloader.
        self.prepare_local_train_loader()
        if self.conf.graph.on_cuda: 
            self.model = self.model.cuda()
            self.last_local_model = self.last_local_model.cuda()
            self.global_model = self.global_model.cuda()

        self.optimizer = create_optimizer.define_optimizer(
            self.conf, model=self.model, optimizer_name=self.conf.optimizer
        )
        self.scheduler = create_scheduler.Scheduler(self.conf, optimizer=self.optimizer)
        self.tracker = RuntimeTracker(metrics_to_track=self.metrics.metric_names)
        self.conf.logger.log(
            f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) enters the local training phase (current communication rounds={self.conf.graph.comm_round})."
        )

        # entering local updates and will finish only after reaching the expected local_n_epochs.
        while True:
            for pos1, pos2, _input, _target in self.train_loader:
                # load data
                with self.timer("load_data", epoch=self.scheduler.epoch_):
                    data_batch = create_dataset.load_data_batch_con(
                        self.conf, pos1, pos2, _input, _target, is_training=True
                    )

                # inference and get current performance.
                with self.timer("forward_pass", epoch=self.scheduler.epoch_):
                    self.optimizer.zero_grad()
                    loss, _ = self._local_training_with_last_local_model(data_batch)

                with self.timer("backward_pass", epoch=self.scheduler.epoch_):
                    loss.backward() #更新本地模型
                    self.optimizer.step()
                    self.scheduler.step()

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

    def _local_training_with_last_local_model(self, data_batch): 
        loss, output = self._inference(data_batch)
        # feature_stu = self.model.activations[-1]
        performance = self.metrics.evaluate(loss, output, data_batch["target"])

        # _, _, last_local_logit = self.last_local_model(data_batch["input"])#model 的输出都要小心
        # # feature_teacher = self.last_local_model.activations[-1]
        # loss2 = self.conf.lamda * self._divergence( #λ为论文中的知识蒸馏的超参
        #     student_logits = output,
        #     teacher_logits = last_local_logit,
        #     KL_temperature=self.conf.KL_T,
        # )

        # loss = loss + loss2 #总损失
        # update tracker.
        if self.tracker is not None:
            self.tracker.update_metrics(
                [loss.item()] + performance + [0], n_samples=data_batch["input"].size(0)
            )
        return loss, output    
   
    def _terminate_comm_round(self):
        # if self.conf.graph.comm_round > 1:
        #     if self.check_overfit:
        #         self._check_overfit()
        #     else:
        #         self.model.load_state_dict(self.received_models[-1].state_dict())
        self.model = self.model.cpu()
        if hasattr(self, 'init_model'):
            del self.init_model
        self.scheduler.clean()
        self.conf.logger.save_json()
        self.conf.logger.log(
            f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) finished one round of federated learning: (comm_round={self.conf.graph.comm_round})."
        )

def sigmoid_rampup(current, rampup_length = 3):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def linear_rampup(current, rampup_length = 15):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length

def attention(x):
    """
    Taken from https://github.com/szagoruyko/attention-transfer
    :param x = activations
    """
    return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))

def attention_diff(x, y):
    """
    Taken from https://github.com/szagoruyko/attention-transfer
    :param x = activations
    :param y = activations
    """
    return (attention(x) - attention(y)).pow(2).mean()
