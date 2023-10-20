# -*- coding: utf-8 -*-

from pcode.workers.worker_base import WorkerBase
from pcode.optimizers.pFedMeOptimizer import pFedMeOptimizer
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
import copy
class WorkerPFedMe(WorkerBase):
    #self.model 为 w，self.theta 为 θ
    def __init__(self, conf):
        super().__init__(conf)
        self.K = conf.K
        #这里假设没有使用异构的模型
        self.arch, self.model = create_model.define_model(
            self.conf, to_consistent_model=False, client_id=1
        )
        self.theta = copy.deepcopy(self.model)
        self.model = copy.deepcopy(list(self.model.parameters()))
    
    def _listen_to_master(self):
        # listen to master, related to the function `_activate_selected_clients` in `master.py`.
        msg = torch.zeros((3, self.conf.n_participated))
        dist.broadcast(tensor=msg, src=0)
        self.conf.graph.client_id, self.conf.graph.comm_round, self.n_local_epochs = (
            msg[:, self.conf.graph.rank - 1].to(int).cpu().numpy().tolist()
        )

        # once we receive the signal, we init for the local training.
        self.model_state_dict = self.theta.state_dict()
        self.model_tb = TensorBuffer(list(self.model_state_dict.values()))
        self.metrics = create_metrics.Metrics(self.model, task="classification")
        dist.barrier()
    
    def _recv_model_from_master(self):
        # related to the function `_send_model_to_selected_clients` in `master.py`
        old_buffer = copy.deepcopy(self.model_tb.buffer)#通过计算模型参数的范数来判断模型是否更新
        dist.recv(self.model_tb.buffer, src=0)
        new_buffer = copy.deepcopy(self.model_tb.buffer)
        self.model_tb.unpack(self.model_state_dict.values())
        self.theta.load_state_dict(self.model_state_dict)
        # θ 复制到 w
        for new_param, localweight in zip(self.theta.parameters(), self.model):
            localweight.data = new_param.data.clone()
        #这里
        self.conf.logger.log(
            f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) received the global/personal model ({self.arch}) from Master. The model status {'is updated' if old_buffer.norm() != new_buffer.norm() else 'is not updated'}."
        )        
        dist.barrier() 
    
    def _send_model_to_master(self):
        dist.barrier()
        self.conf.logger.log(
            f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) sending the model ({self.arch}) back to Master."
        )
        flatten_model = TensorBuffer(list(self.theta.state_dict().values()))
        dist.send(tensor=flatten_model.buffer, dst=0)
        dist.barrier()
        
    def update_parameters(self, new_params):
        for param , new_param in zip(self.theta.parameters(), new_params):
            param.data = new_param.data.clone()      
        
    def _train(self):
        self.theta.train()
        # init the model and dataloader.
        self.prepare_local_train_loader()
        # 将 model 中元素存入 cuda
        model_on_cuda = []
        if self.conf.graph.on_cuda:
            self.theta = self.theta.cuda()
            for data in self.model:
                model_on_cuda.append(data.cuda())
            self.model = model_on_cuda
        
        # define optimizer, scheduler and runtime tracker.
        self.optimizer = pFedMeOptimizer(self.theta.parameters(), lr=self.conf.lr, lamda=self.conf.pFedMeLamda)
        self.scheduler = create_scheduler.Scheduler(self.conf, optimizer=self.optimizer)
        self.tracker = RuntimeTracker(metrics_to_track=self.metrics.metric_names)
        self.conf.logger.log(
            f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) enters the local training phase (current communication rounds={self.conf.graph.comm_round})."
        )

        # entering local updates and will finish only after reaching the expected local_n_epochs.
        while True:
            for _input, _target in self.train_loader:
                # load data
                with self.timer("load_data", epoch=self.scheduler.epoch_):
                    data_batch = create_dataset.load_data_batch(
                        self.conf, _input, _target, is_training=True
                    )

                for i in range(self.K):
                    self.optimizer.zero_grad()
                    output = self.theta(data_batch["input"])
                    loss = self.criterion(output, data_batch["target"])
                    loss.backward()
                    self.personalized_model_bar, _ = self.optimizer.step(self.model)
                
                # update local weight after finding aproximate theta, 这里其实在更新 local model了,即更新 w
                for new_param, localweight in zip(self.personalized_model_bar, self.model):
                    localweight.data = localweight.data - self.conf.pFedMeLamda* self.conf.lr * (localweight.data - new_param.data)
                
                self.update_parameters(self.model)
                #eval
                performance = self.metrics.evaluate(loss, output, data_batch["target"])

                # update tracker.
                if self.tracker is not None:
                    self.tracker.update_metrics(
                        [loss.item()] + performance, n_samples=data_batch["input"].size(0)
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



    