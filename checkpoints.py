import os
import torch
from config import params

def save_model(net, file_name):
    '''Save trained model'''
    if not os.path.exists(params.model_root):
        os.makedirs(params.model_root)
    torch.save(net.state_dict(),
               os.path.join(params.model_root, file_name))
    print("[*]Save trained model to: {}".format(os.path.join(params.model_root, file_name)))


def restore_model(net, trained_model_path):
    '''Load trained model'''
    if trained_model_path and os.path.exists(trained_model_path):
        net.load_state_dict(torch.load(trained_model_path))
        print('[*]Restore model from {}'.format(os.path.abspath(trained_model_path)))
    return net

