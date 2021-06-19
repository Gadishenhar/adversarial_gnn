import os
import torch.optim as optim
import adabound


def getGitPath():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    git_dir = os.path.dirname(os.path.dirname(current_dir))
    return git_dir


def create_optimizer(params, mode, *args, **kwargs):
    if mode == 'SGD':
        opt = optim.SGD(params, *args, momentum=0., **kwargs)
    elif mode.startswith('nesterov'):
        momentum = float(mode[len('nesterov'):])
        opt = optim.SGD(params, *args, momentum=momentum, nesterov=True, **kwargs)
    elif mode.lower() == 'adam':
        betas = kwargs.pop('betas', (.9, .999))
        opt = optim.Adam(params, *args, betas=betas, amsgrad=True,
                         weight_decay=1e-4, **kwargs)
    elif mode.lower() == 'adam_hyp2':
        betas = kwargs.pop('betas', (.5, .99))
        opt = optim.Adam(params, *args, betas=betas, amsgrad=True, **kwargs)
    elif mode.lower() == 'adam_hyp3':
        betas = kwargs.pop('betas', (0., .99))
        opt = optim.Adam(params, *args, betas=betas, amsgrad=True, **kwargs)
    elif mode.lower() == 'adam_sparse':
        betas = kwargs.pop('betas', (.9, .999))
        opt = optim.SparseAdam(params, *args, weight_decay=1e-4, betas=betas)
    elif mode.lower() == 'adam_sparse_hyp2':
        betas = kwargs.pop('betas', (.5, .99))
        opt = optim.SparseAdam(params, *args, betas=betas)
    elif mode.lower() == 'adam_sparse_hyp3':
        betas = kwargs.pop('betas', (.0, .99))
        opt = optim.SparseAdam(params, *args, betas=betas)
    elif mode.lower() == 'adabound':
        opt = adabound.AdaBound(params, *args, final_lr=0.1)
    else:
        raise NotImplementedError()
    return opt
