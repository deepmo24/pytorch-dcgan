import os
import torch
import config

def save_model(net, file_name):
    '''Save trained model'''
    if not os.path.exists(config.model_root):
        os.makedirs(config.model_root)
    torch.save(net.state_dict(),
               os.path.join(config.model_root, file_name))
    print("[*]Save trained model to: {}".format(os.path.join(config.model_root, file_name)))


def restore_model(net, trained_model_path):
    if trained_model_path and os.path.exists(trained_model_path):
        net.load_state_dict(torch.load(trained_model_path))
        print('[*]Restore model from {}'.format(os.path.abspath(trained_model_path)))
    return net

