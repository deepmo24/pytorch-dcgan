import os
import random
import torch
from config import params
from datasets.mnist import get_mnist
from datasets.celeba import get_celeba


def init_random_seed(manual_seed):
    '''Init random seed.'''
    seed = None
    if manual_seed is None:
        seed = random.randint(1, 2**32-2)
    else:
        seed = manual_seed
    print("use random seed: {}".format(seed))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def check_dirs():
    '''check the needed dirs of params'''
    if not os.path.exists(params.model_root):
        os.makedirs(params.model_root)
    if not os.path.exists(params.samples_root):
        os.makedirs(params.samples_root)
    if not os.path.exists(params.save_root):
        os.makedirs(params.save_root)
    if not os.path.exists(params.dataset_root):
        os.makedirs(params.dataset_root)



def get_data_loader(name):
    '''Get data loader by name'''
    if name == 'mnist':
        return get_mnist()
    elif name == 'celeba':
        return get_celeba()
    else:
        assert False, '[*] dataset not implement!'