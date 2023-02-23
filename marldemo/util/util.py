import copy
import numpy as np
import math
import torch
import torch.nn as nn
import os
from pathlib import Path
from marldemo.env.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv
import json
from tensorboardX import SummaryWriter
import random
from marldemo.env.pettingzoo_mpe.pettingzoo_mpe_env import PettingZooMPEEnv

def _t2n(x):
    return x.detach().cpu().numpy()


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def check(input):
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output


def get_grad_norm(it):
    sum_grad = 0
    for x in it:
        if x.grad is None:
            continue
        sum_grad += x.grad.norm() ** 2
    return math.sqrt(sum_grad)


def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (e > d).float()
    return a*e**2/2 + b*d*(abs(e)-d/2)


def mse_loss(e):
    return e**2/2


def get_shape_from_act_space(act_space):
    if act_space.__class__.__name__ == 'Discrete':
        act_shape = 1
    elif act_space.__class__.__name__ == "Box":
        act_shape = act_space.shape[0]
    return act_shape


def make_train_env(seed, n_threads, env_args):
    """Make env for training."""
    def get_env_fn(rank):
        def init_env():
            assert env_args["scenario"] in ["simple_v2", "simple_spread_v2", "simple_reference_v2",
                                            "simple_speaker_listener_v3"], "only cooperative scenarios in MPE are supported"
            env = PettingZooMPEEnv(env_args)
            env.seed(seed + rank * 1000)
            return env
        return init_env

    if n_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(n_threads)])


def make_eval_env(seed, n_threads, env_args):
    """Make env for evaluation."""
    def get_env_fn(rank):
        def init_env():
            env = PettingZooMPEEnv(env_args)
            env.seed(seed * 50000 + rank * 10000)
            return env
        return init_env

    if n_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(n_threads)])

def make_render_env(seed, env_args):
    env = PettingZooMPEEnv({**env_args, "render_mode": "human"})
    env.seed(seed * 60000)
    return env

def seed(args):
    """Seed the program."""
    if not args["seed_specify"]:
        args["seed"] = np.random.randint(1000, 10000)
    random.seed(args["seed"])
    np.random.seed(args["seed"])
    os.environ['PYTHONHASHSEED'] = str(args["seed"])
    torch.manual_seed(args["seed"])
    torch.cuda.manual_seed(args["seed"])
    torch.cuda.manual_seed_all(args["seed"])


def init_device(args):
    """Init device."""
    if args["cuda"] and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        if args["cuda_deterministic"]:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
    return device


def init_dir(env, env_args, algo, exp_name, seed):
    """Init directory for saving results."""
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
            0] + "/results") / env / env_args["scenario"] / algo / exp_name / str(seed)
    if not run_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = run_dir / curr_run
    os.makedirs(str(run_dir))
    log_dir = str(run_dir / 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writter = SummaryWriter(log_dir)
    save_dir = str(run_dir / 'models')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return run_dir, log_dir, save_dir, writter


def save_config(args, algo_args, env_args, run_dir):
    """Save the configuration of the program."""
    main_args_dict = args.__dict__
    all_args = {"main_args": main_args_dict,
                "algo_args": algo_args,
                "env_args": env_args}
    config_dir = run_dir / 'config.json'
    with open(config_dir, 'w') as f:
        json.dump(all_args, f)


def get_init_method(initialization_method):
    return nn.init.__dict__[initialization_method]