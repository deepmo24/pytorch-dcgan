import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import config
from models.dcgan import Discriminator, Generator
from datasets.celeba import get_celeba
from trainer import Trainer
from checkpoints import restore_model
from utils import check_dirs

def main():

    #check the needed dirs of config
    check_dirs()

    cudnn.benchmark = True
    torch.cuda.set_device(config.gpu_id[0]) #set current device

    print('=== Build model ===')
    #gpu mode
    generator = Generator()
    discriminator = Discriminator()
    generator = nn.DataParallel(generator, device_ids=config.gpu_id).cuda()
    discriminator = nn.DataParallel(discriminator, device_ids=config.gpu_id).cuda()

    # restore trained model
    if config.generator_restored:
        generator = restore_model(generator, config.generator_restored)
    if config.discriminator_resotred:
        discriminator = restore_model(discriminator, config.discriminator_resotred)


    #data loader
    print('=== Load data ===')
    train_dataloader = get_celeba()

    # container of training
    trainer = Trainer(generator,discriminator)

    if config.mode == 'train':
        print('=== Begin training ===')
        trainer.train(train_dataloader)
        print('=== Generate {} images, saving in {} ==='.format(config.num_images, config.save_root))
        trainer.generate_images(config.num_images, config.save_root)
    elif config.mode == 'test':
        if config.generator_restored:
            print('=== Generate {} images, saving in {} ==='.format(config.num_images, config.save_root))
            trainer.generate_images(config.num_images, config.save_root)
        else:
            assert False, '[*]load Generator model first!'

    else:
        assert False, "[*]mode must be 'train' or 'test'!"





if __name__ == '__main__':
    main()