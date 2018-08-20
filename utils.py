import config
import os


def check_dirs():
    '''check the needed dirs of config'''
    if not os.path.exists(config.model_root):
        os.makedirs(config.model_root)
    if not os.path.exists(config.samples_root):
        os.makedirs(config.samples_root)
    if not os.path.exists(config.save_root):
        os.makedirs(config.save_root)

    if not os.path.exists(config.dataset_root):
        assert False, '[*]dataset root not exist!'



