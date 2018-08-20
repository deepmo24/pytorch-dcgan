import torch
import torch.nn as nn
import torch.optim as optim
import config
from checkpoints import save_model
import torchvision.utils as vutils
import os

class Trainer(object):
    '''Trainer class which contains the training environment'''
    def __init__(self, G, D):
        self.G = G
        self.D = D

        # criterion function
        self.criterion = nn.BCELoss()

        # optimizer
        self.optimizer_G = optim.Adam(G.parameters(),
                                      lr=config.g_lr,
                                      betas=config.betas)
        self.optimizer_D = optim.Adam(D.parameters(),
                                      lr=config.d_lr,
                                      betas=config.betas)


    def train(self, train_loader):
        '''training process'''

        # set train state for Dropout and BN layers
        self.G.train()
        self.D.train()


        sample_z_batch = torch.randn(config.batch_size, config.z_dim).cuda() # fix noise

        len_data_loader = len(train_loader)
        for epoch in range(config.max_epoch):
            for step, (images,_) in enumerate(train_loader):

                #######################
                # (1)update D network #
                #######################

                self.optimizer_D.zero_grad()

                images = images.cuda()
                z = torch.randn(images.size(0), config.z_dim).cuda()

                predict_real = self.D(images)
                probability_real = predict_real.mean().item()
                fake_images = self.G(z)
                predict_fake = self.D(fake_images.detach())
                probability_fake = predict_fake.mean().item()

                d_loss_real = self.criterion(predict_real, torch.ones(images.size(0)).cuda())
                d_loss_fake = self.criterion(predict_fake, torch.zeros(images.size(0)).cuda())

                d_loss = d_loss_real + d_loss_fake

                d_loss.backward()
                self.optimizer_D.step()


                #######################
                # (2)update G network #
                #######################

                self.optimizer_G.zero_grad()

                predict_fake = self.D(fake_images)
                g_loss = self.criterion(predict_fake, torch.ones(images.size(0)).cuda())

                g_loss.backward()
                self.optimizer_G.step()



                # log information
                if (step+1) % config.log_step == 0:
                    print('Epoch [{}/{}] Step [{}/{}]: '
                          'd_loss={:.5f} g_loss={:.5f} real_avgP={:.4f} fake_avgP={:.4f}'
                          .format(epoch+1,
                                  config.max_epoch,
                                  step+1,
                                  len_data_loader,
                                  d_loss,
                                  g_loss,
                                  probability_real,
                                  probability_fake))

                # save samples after every 1/20 iterations
                if (step+1) % (len_data_loader//20) == 0:
                    samples = self.G(sample_z_batch)
                    samples = samples.detach().cpu()
                    vutils.save_image(samples,
                                      os.path.join(config.samples_root,'train_{:02d}_{:04d}.png'.format(epoch+1,step+1)),
                                      normalize=True)
                    print('[*]save samples in',os.path.join(config.samples_root,'train_{:02d}_{:04d}.png'.format(epoch+1,step+1)))


            #save model
            if epoch % config.save_epoch == 0:
                save_model(self.G, 'Generator-{}.pt'.format(epoch+1))
                save_model(self.D, 'Discriminator-{}.pt'.format(epoch+1))

        #save final model
        save_model(self.G, 'Generator-final.pt')
        save_model(self.D, 'Discriminator-final.pt')



    def test(self, test_loader):
        pass

    def generate_images(self, N, save_root):
        '''generate N(or less) samples'''
        if not os.path.exists(save_root):
            os.makedirs(save_root)

        self.G.eval()

        iterations = N // config.batch_size
        count = 0
        for i in range(iterations):
            z = torch.randn(config.batch_size, config.z_dim).cuda()
            samples = self.G(z)
            samples = samples.detach().cpu()
            for image in samples:
                count += 1
                vutils.save_image(image,
                                  os.path.join(save_root,'image_{:04d}.png'.format(count)),
                                  normalize=True)