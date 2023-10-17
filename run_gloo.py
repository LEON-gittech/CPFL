import os
import random
from datetime import timedelta

# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import pcode.utils.checkpoint as checkpoint
import pcode.utils.logging as logging
import pcode.utils.param_parser as param_parser
import pcode.utils.topology as topology
from parameters import get_args
from pcode.masters.master_fedavg import MasterFedAvg
# from pcode.masters.master_fedfomo import MasterFedFomo
# from pcode.masters.master_fedper import MasterFedPer
# from pcode.masters.master_lg_fedavg import MasterLGFedAvg
# from pcode.masters.master_local_train import MasterLocalTrain
from pcode.masters.master_pFedSD import MasterpFedSD
# from pcode.masters.master_pFedMe import MasterpFedMe
# from pcode.masters.master_tlkt import MasterTLKT
from pcode.workers.worker_fedavg import WorkerFedAvg
# from pcode.workers.worker_fedfomo import WorkerFedFomo
# from pcode.workers.worker_fedper import WorkerFedPer
# from pcode.workers.worker_lg_fedavg import WorkerLGFedAvg
# from pcode.workers.worker_local_train import WorkerLocalTrain
from pcode.workers.worker_pFedSD import WorkerpFedSD
#对比学习
## SD+Con
from pcode.masters.master_pFedSD_con import MasterpFedSDCon
from pcode.workers.worker_pFedSD_con import WorkerpFedSDCon
## Con
from pcode.masters.master_fedavg_con import MasterFedAvgCon
from pcode.workers.worker_fedavg_con import WorkerFedAvgCon
# Moon
from pcode.masters.master_fedavg_moon import MasterFedAvgMoon
from pcode.workers.worker_fedavg_moon import WorkerFedAvgMoon
# Con+Moon
from pcode.masters.master_fedavg_con_moon import MasterFedAvgConMoon
from pcode.workers.worker_fedavg_con_moon import WorkerFedAvgConMoon
# from pcode.workers.worker_pFedme import WorkerpFedMe
# from pcode.workers.worker_tlkt import WorkerTLKT


def main(rank,size,conf,port):
    # init the distributed world.
    try:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = port
        dist.init_process_group("gloo",rank=rank,world_size=size, timeout=timedelta(minutes=300))
    except AttributeError as e:
        print(f"failed to init the distributed world: {e}.")
        conf.distributed = False

    # init the config.
    init_config(conf) #同步

    if conf.algo == "fedavg":
        master_func = MasterFedAvg
        worker_func = WorkerFedAvg
    # elif conf.algo == "fedprox":
    #     master_func = MasterFedAvg
    #     worker_func = WorkerFedAvg
    # elif conf.algo == "fedper":
    #     master_func = MasterFedPer
    #     worker_func = WorkerFedPer
    # elif conf.algo == "lg_fedavg":
    #     master_func = MasterLGFedAvg
    #     worker_func = WorkerLGFedAvg
    # elif conf.algo == "pFedme":
    #     master_func = MasterpFedMe
    #     worker_func = WorkerpFedMe    
    # elif conf.algo == "fedfomo":
    #     master_func = MasterFedFomo
    #     worker_func = WorkerFedFomo
    # elif conf.algo == "local_training":
    #     master_func = MasterLocalTrain
    #     worker_func = WorkerLocalTrain
    # elif conf.algo == "tlkt":
    #     master_func = MasterTLKT
    #     worker_func = WorkerTLKT
    elif conf.algo == "pFedSD":
        master_func = MasterpFedSD
        worker_func = WorkerpFedSD
    #对比学习
    elif conf.algo == "pFedSD_con":
        master_func = MasterpFedSDCon
        worker_func = WorkerpFedSDCon
    elif conf.algo == "fedavg_con":
        master_func = MasterFedAvgCon
        worker_func = WorkerFedAvgCon
    elif conf.algo == "fedavg_moon":
        master_func = MasterFedAvgMoon
        worker_func = WorkerFedAvgMoon
    elif conf.algo == "fedavg_con_moon":
        master_func = MasterFedAvgConMoon
        worker_func = WorkerFedAvgConMoon

    else:
        raise NotImplementedError

    # start federated learning.
    process = master_func(conf) if conf.graph.rank == 0 else worker_func(conf)
    process.run()


def init_config(conf):
    # define the graph for the computation.
    conf.graph = topology.define_graph_topology(
        world=conf.world,
        world_conf=conf.world_conf,
        n_participated=conf.n_participated,
        on_cuda=conf.on_cuda,
    )
    conf.graph.rank = dist.get_rank()

    # init related to randomness on cpu.
    """_summary_
    """    
    if not conf.same_seed_process:
        conf.manual_seed = 1000 * conf.manual_seed + conf.graph.rank
    np.random.seed(conf.manual_seed)
    conf.random_state = np.random.RandomState(conf.manual_seed)
    torch.manual_seed(conf.manual_seed)
    random.seed(conf.manual_seed)

    # configure cuda related.
    if conf.graph.on_cuda:
        assert torch.cuda.is_available()
        torch.cuda.manual_seed(conf.manual_seed)
        # torch.cuda.set_device(conf.graph.primary_device)
        # device_id = conf.graph.rank % torch.cuda.device_count()
        torch.cuda.set_device(conf.graph.rank % torch.cuda.device_count()) #在这里对数据进行了分布式
        # print(torch.cuda.current_device())
        torch.backends.cudnn.enabled = True
        # torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.deterministic = True if conf.train_fast else False
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(conf.manual_seed)

    # init the model arch info.
    conf.arch_info = (
        param_parser.dict_parser(conf.complex_arch)
        if conf.complex_arch is not None
        else {"master": conf.arch, "worker": conf.arch}
    )
    conf.arch_info["worker"] = conf.arch_info["worker"].split(":")

    # parse the fl_aggregate scheme.
    conf._fl_aggregate = conf.fl_aggregate #默认为 federated avg
    conf.fl_aggregate = (
        param_parser.dict_parser(conf.fl_aggregate)
        if conf.fl_aggregate is not None
        else conf.fl_aggregate
    )
    [setattr(conf, f"fl_aggregate_{k}", v) for k, v in conf.fl_aggregate.items()]

    # define checkpoint for logging (for federated learning server).
    checkpoint.init_checkpoint(conf, rank=str(conf.graph.rank)) #建立 checkpoint 的子目录 0,1,2 。。。。

    # configure logger.
    conf.logger = logging.Logger(conf.checkpoint_dir) #同上

    # display the arguments' info.
    if conf.graph.rank == 0:
        logging.display_args(conf)

    # sync the processes.
    dist.barrier()


import time

if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
    conf = get_args() #如果不在调试，应注释掉加载参数文件的代码
    
    import json
    with open('args.txt','w') as f:
        json.dump(conf.__dict__,f,indent=2)
    
    #client 的参与率
    conf.n_participated = int(conf.n_clients * conf.participation_ratio + 0.5)
    # conf.timestamp = str(int(time.time()))
    size = conf.n_participated + 1
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=main, args=(rank, size,conf,conf.port))
        p.start()
        print("rank {} started",rank)
        processes.append(p)

    for p in processes:
        p.join()

