import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from config import params
from models.dcgan import Discriminator, Generator
from trainer import Trainer
from checkpoints import restore_model
from utils import init_random_seed, check_dirs, get_data_loader

def main():

    # init random seed
    init_random_seed(params.manual_seed)
    #check the needed dirs of config
    check_dirs()

    cudnn.benchmark = True
    torch.cuda.set_device(params.gpu_id[0]) #set current device

    print('=== Build model ===')
    #gpu mode
    generator = Generator()
    discriminator = Discriminator()
    generator = nn.DataParallel(generator, device_ids=params.gpu_id).cuda()
    discriminator = nn.DataParallel(discriminator, device_ids=params.gpu_id).cuda()

    # restore trained model
    if params.generator_restored:
        generator = restore_model(generator, params.generator_restored)
    if params.discriminator_restored:
        discriminator = restore_model(discriminator, params.discriminator_restored)

    # container of training
    trainer = Trainer(generator,discriminator)

    if params.mode == 'train':
        # data loader
        print('=== Load data ===')
        train_dataloader = get_data_loader(params.dataset)

        print('=== Begin training ===')
        trainer.train(train_dataloader)
        print('=== Generate {} images, saving in {} ==='.format(params.num_images, params.save_root))
        trainer.generate_images(params.num_images, params.save_root)
    elif params.mode == 'test':
        if params.generator_restored:
            print('=== Generate {} images, saving in {} ==='.format(params.num_images, params.save_root))
            trainer.generate_images(params.num_images, params.save_root)
        else:
            assert False, '[*]load Generator model first!'

    else:
        assert False, "[*]mode must be 'train' or 'test'!"



if __name__ == '__main__':
    main()