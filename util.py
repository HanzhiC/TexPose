import numpy as np
import os, sys, time
import shutil
import datetime
import torch
import torch.nn.functional as torch_F
import ipdb
import types
import termcolor
import socket
import contextlib
from easydict import EasyDict as edict


# convert to colored strings
def red(message, **kwargs): return termcolor.colored(str(message), color="red",
                                                     attrs=[k for k, v in kwargs.items() if v is True])


def green(message, **kwargs): return termcolor.colored(str(message), color="green",
                                                       attrs=[k for k, v in kwargs.items() if v is True])


def blue(message, **kwargs): return termcolor.colored(str(message), color="blue",
                                                      attrs=[k for k, v in kwargs.items() if v is True])


def cyan(message, **kwargs): return termcolor.colored(str(message), color="cyan",
                                                      attrs=[k for k, v in kwargs.items() if v is True])


def yellow(message, **kwargs): return termcolor.colored(str(message), color="yellow",
                                                        attrs=[k for k, v in kwargs.items() if v is True])


def magenta(message, **kwargs): return termcolor.colored(str(message), color="magenta",
                                                         attrs=[k for k, v in kwargs.items() if v is True])


def grey(message, **kwargs): return termcolor.colored(str(message), color="grey",
                                                      attrs=[k for k, v in kwargs.items() if v is True])


def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


def get_time(sec):
    d = int(sec // (24 * 60 * 60))
    h = int(sec // (60 * 60) % 24)
    m = int((sec // 60) % 60)
    s = int(sec % 60)
    return d, h, m, s


def add_datetime(func):
    def wrapper(*args, **kwargs):
        datetime_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(grey("[{}] ".format(datetime_str), bold=True), end="")
        return func(*args, **kwargs)

    return wrapper


def add_functionname(func):
    def wrapper(*args, **kwargs):
        print(grey("[{}] ".format(func.__name__), bold=True))
        return func(*args, **kwargs)

    return wrapper


def pre_post_actions(pre=None, post=None):
    def func_decorator(func):
        def wrapper(*args, **kwargs):
            if pre: pre()
            retval = func(*args, **kwargs)
            if post: post()
            return retval

        return wrapper

    return func_decorator


debug = ipdb.set_trace


class Log:
    def __init__(self):
        pass

    def process(self, pid):
        print(grey("Process ID: {}".format(pid), bold=True))

    def title(self, message):
        print(yellow(message, bold=True, underline=True))

    def info(self, message):
        print(magenta(message, bold=True))

    def options(self, opt, level=0):
        for key, value in sorted(opt.items()):
            if isinstance(value, (dict, edict)):
                print("   " * level + cyan("* ") + green(key) + ":")
                self.options(value, level + 1)
            else:
                print("   " * level + cyan("* ") + green(key) + ":", yellow(value))

    def loss_train(self, opt, ep, lr, loss, timer):
        if not opt.max_epoch: return
        message = grey("[train] ", bold=True)
        message += "epoch {}/{}".format(cyan(ep, bold=True), opt.max_epoch)
        message += ", lr:{}".format(yellow("{:.2e}".format(lr), bold=True))
        message += ", loss:{}".format(red("{:.3e}".format(loss), bold=True))
        message += ", time:{}".format(blue("{0}-{1:02d}:{2:02d}:{3:02d}".format(*get_time(timer.elapsed)), bold=True))
        message += " (ETA:{})".format(blue("{0}-{1:02d}:{2:02d}:{3:02d}".format(*get_time(timer.arrival))))
        print(message)

    def loss_val(self, opt, loss):
        message = grey("[val] ", bold=True)
        message += "loss:{}".format(red("{:.3e}".format(loss), bold=True))
        print(message)


log = Log()


def update_timer(opt, timer, ep, it_per_ep):
    if not opt.max_epoch: return
    momentum = 0.99
    timer.elapsed = time.time() - timer.start
    timer.it = timer.it_end - timer.it_start
    # compute speed with moving average
    timer.it_mean = timer.it_mean * momentum + timer.it * (1 - momentum) if timer.it_mean is not None else timer.it
    timer.arrival = timer.it_mean * it_per_ep * (opt.max_epoch - ep)


# move tensors to device in-place
def move_to_device(X, device):
    if isinstance(X, dict):
        for k, v in X.items():
            X[k] = move_to_device(v, device)
    elif isinstance(X, list):
        for i, e in enumerate(X):
            X[i] = move_to_device(e, device)
    elif isinstance(X, tuple) and hasattr(X, "_fields"):  # collections.namedtuple
        dd = X._asdict()
        dd = move_to_device(dd, device)
        return type(X)(**dd)
    elif isinstance(X, torch.Tensor):
        return X.to(device=device)
    return X


def to_dict(D, dict_type=dict):
    D = dict_type(D)
    for k, v in D.items():
        if isinstance(v, dict):
            D[k] = to_dict(v, dict_type)
    return D


def get_child_state_dict(state_dict, key):
    return {".".join(k.split(".")[1:]): v for k, v in state_dict.items() if k.startswith("{}.".format(key))}


def restore_checkpoint(opt, model, load_name=None, resume=False):
    assert ((load_name is None) == (resume is not False))  # resume can be True/False or epoch numbers
    if resume:
        load_name = "{0}/model.ckpt".format(opt.output_path) if resume is True else \
            "{0}/model/{1}.ckpt".format(opt.output_path, resume)
    checkpoint = torch.load(load_name, map_location=opt.device)

    # load individual (possibly partial) children modules
    for name, child in model.graph.named_children():
        child_state_dict = get_child_state_dict(checkpoint["graph"], name)
        if child_state_dict:
            print("restoring {}...".format(name))
            child.load_state_dict(child_state_dict)

    # Load optimizer and scheduler
    for key in model.__dict__:
        if key.split("_")[0] in ["optim", "sched"] and key in checkpoint and resume:
            print("restoring {}...".format(key))
            getattr(model, key).load_state_dict(checkpoint[key])

    if resume:
        ep, it = checkpoint["epoch"], checkpoint["iter"]
        if resume is not True:  # not boolean
            assert (resume == ep) or (resume == it)
        print("resuming from epoch {0} (iteration {1})".format(ep, it))
    else:
        ep, it = None, None
    return ep, it


def restore_pretrain_partial_checkpoint(opt, model, load_name=None, resume=False):
    assert ((load_name is None) == (resume is not False))  # resume can be True/False or epoch numbers
    if resume:
        load_name = "{0}/{1}/pretrain_model.ckpt".format(opt.output_root, opt.group) if resume is True else \
            "{0}/model/{1}.ckpt".format(opt.output_path, resume)
    checkpoint = torch.load(load_name, map_location=opt.device)

    # load individual (possibly partial) children modules
    for name, child in model.graph.named_children():
        child_state_dict = {".".join(k.split(".")[1:]): v for k, v in checkpoint["graph"].items()
                            if 'mlp_feat' in k and k.startswith("{}.".format(name))}
        # child_state_dict = get_child_state_dict(checkpoint["graph"], name)

        if child_state_dict:
            print("restoring pretrained {} mlp_feat for geometric feature extraction...".format(name))
            model_dict = child.state_dict()
            pretrained_dict = {k: v for k, v in child_state_dict.items() if k in child.state_dict()}
            model_dict.update(pretrained_dict)
            child.load_state_dict(model_dict)
    ep, it = None, None
    return ep, it


def restore_pretrain_nerf(opt, model, load_name=None, resume=False):
    assert ((load_name is None) == (resume is not False))  # resume can be True/False or epoch numbers
    if resume:
        load_name = "{0}/pretrain_model_real.ckpt".format(opt.output_path) if resume is True else \
            "{0}/model/{1}.ckpt".format(opt.output_path, resume)
    checkpoint = torch.load(load_name, map_location=opt.device)

    # load individual (possibly partial) children modules
    for name, child in model.graph.named_children():
        if name == 'nerf':
            child_state_dict = get_child_state_dict(checkpoint["graph"], name)
            if child_state_dict:
                print("restoring {} NeRF model from real pretrain...".format(name))
                child.load_state_dict(child_state_dict)
        else:
            continue
    ep, it = None, None
    return ep, it

def save_checkpoint(opt, model, ep, it, latest=False, children=None):
    os.makedirs("{0}/model".format(opt.output_path), exist_ok=True)
    if children is not None:
        graph_state_dict = {k: v for k, v in model.graph.state_dict().items() if k.startswith(children)}
    else:
        graph_state_dict = model.graph.state_dict()
    checkpoint = dict(
        epoch=ep,
        iter=it,
        graph=graph_state_dict,
    )
    for key in model.__dict__:
        if key.split("_")[0] in ["optim", "sched"]:
            checkpoint.update({key: getattr(model, key).state_dict()})
    torch.save(checkpoint, "{0}/model.ckpt".format(opt.output_path))
    if not latest:
        # shutil.copy("{0}/model.ckpt".format(opt.output_path),
        #             "{0}/model/{1}.ckpt".format(opt.output_path, ep or it))  # if ep is None, track it instead
        shutil.copy("{0}/model.ckpt".format(opt.output_path),
                    "{0}/model/{1}.ckpt".format(opt.output_path, it))  # if ep is None, track it instead

def check_socket_open(hostname, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    is_open = False
    try:
        s.bind((hostname, port))
    except socket.error:
        is_open = True
    finally:
        s.close()
    return is_open


def get_layer_dims(layers):
    # return a list of tuples (k_in,k_out)
    return list(zip(layers[:-1], layers[1:]))


@contextlib.contextmanager
def suppress(stdout=False, stderr=False):
    with open(os.devnull, "w") as devnull:
        if stdout: old_stdout, sys.stdout = sys.stdout, devnull
        if stderr: old_stderr, sys.stderr = sys.stderr, devnull
        try:
            yield
        finally:
            if stdout: sys.stdout = old_stdout
            if stderr: sys.stderr = old_stderr


def ray_batch_sample(ray_identity, ray_idx):
    assert ray_identity.shape[0] == ray_idx.shape[0]
    B, HW, _ = ray_identity.shape
    B, N_sample = ray_idx.shape
    ray_identity = ray_identity.view(B * HW, -1)
    samples_cumsum = ray_idx + HW * torch.arange(B).cuda().unsqueeze(1)  # No need to reshape
    ray_identity_sample = ray_identity[samples_cumsum].view(B, N_sample, -1)
    return ray_identity_sample
