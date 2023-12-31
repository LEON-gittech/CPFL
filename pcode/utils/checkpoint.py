# -*- coding: utf-8 -*-
import json
import shutil
import time
from os.path import join

import torch
from pcode.utils.op_files import is_jsonable
from pcode.utils.op_paths import build_dirs


def get_checkpoint_folder_name(conf):
    # get optimizer info.
    optim_info = "{}".format(conf.optimizer)

    # get n_participated
    conf.n_participated = int(conf.n_clients * conf.participation_ratio + 0.5)

    # concat them together.
    return "_algo-{}_lr-{}_n_comm_rounds-{}_local_n_epochs-{}_batchsize-{}_n_clients-{}_participation_ratio-{}".format(
        conf.algo,
        conf.lr,
        conf.n_comm_rounds,
        conf.local_n_epochs,
        conf.batch_size,
        conf.n_clients,
        conf.participation_ratio,
    )


def init_checkpoint(conf, rank=None):
    """
    The function `init_checkpoint` initializes the checkpoint directory for saving models during
    training, creating the necessary directories if they don't exist.
    
    :param conf: The `conf` parameter is an object that contains various configuration settings for the
    checkpoint initialization process. It likely includes information such as the checkpoint directory,
    data directory, architecture, experiment name, and timestamp
    :param rank: The `rank` parameter is an optional argument that represents the rank of the process.
    It is used to create a separate checkpoint directory for each process when running in a distributed
    setting. If `rank` is not provided, the checkpoint directory is created for the main process
    """
    # init checkpoint_root for the main process.
    conf.checkpoint_root = join(
        conf.checkpoint,
        conf.data,
        conf.arch,
        conf.experiment,
        conf.timestamp + get_checkpoint_folder_name(conf),
    )
    if conf.save_some_models is not None:
        conf.save_some_models = conf.save_some_models.split(",")

    if rank is None:
        # if the directory does not exists, create them.
        build_dirs(conf.checkpoint_root)
    else:
        conf.checkpoint_dir = join(conf.checkpoint_root, rank)
        build_dirs(conf.checkpoint_dir)


def _save_to_checkpoint(state, dirname, filename):
    checkpoint_path = join(dirname, filename)
    torch.save(state, checkpoint_path)
    return checkpoint_path


def save_arguments(conf):
    # save the configure file to the checkpoint.
    # write_pickle(conf, path=join(conf.checkpoint_root, "arguments.pickle"))
    with open(join(conf.checkpoint_root, "arguments.json"), "w") as fp:
        json.dump(
            dict(
                [
                    (k, v)
                    for k, v in conf.__dict__.items()
                    if is_jsonable(v) and type(v) is not torch.Tensor
                ]
            ),
            fp,
            indent=" ",
        )


def save_to_checkpoint(conf, state, is_best, dirname, filename, save_all=False):
    # save full state.
    checkpoint_path = _save_to_checkpoint(state, dirname, filename)
    best_model_path = join(dirname, "model_best.pth.tar")
    if is_best:
        shutil.copyfile(checkpoint_path, best_model_path)
    if save_all:
        shutil.copyfile(
            checkpoint_path,
            join(
                dirname, "checkpoint_c_round_%s.pth.tar" % state["current_comm_round"]
            ),
        )
    elif conf.save_some_models is not None:
        if str(state["current_comm_round"]) in conf.save_some_models:
            shutil.copyfile(
                checkpoint_path,
                join(
                    dirname,
                    "checkpoint_c_round_%s.pth.tar" % state["current_comm_round"],
                ),
            )
