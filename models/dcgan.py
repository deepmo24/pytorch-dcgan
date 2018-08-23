'''Discriminator and Generator of DCGAN'''

from torch import nn
from config import params

class Discriminator(nn.Module):
    '''Discriminator'''

    def __init__(self):
        super(Discriminator, self).__init__()

        input_dim = params.c_dim
        df_dim = params.df_dim
        conv_h_size = params.height // 2 ** 4 # image size after 4 conv layers. This requires params.height(width) to be able to divise 2**4.
        conv_w_size = params.width // 2 ** 4

        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_dim, df_dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(df_dim, df_dim*2, 4, 2, 1),
            nn.BatchNorm2d(df_dim*2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(df_dim*2, df_dim*4, 4, 2, 1),
            nn.BatchNorm2d(df_dim*4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(df_dim * 4, df_dim * 8, 4, 2, 1),
            nn.BatchNorm2d(df_dim * 8),
            nn.LeakyReLU(0.2)
        )

        self.output_layer = nn.Sequential(
            nn.Linear(df_dim*8*conv_h_size*conv_w_size, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        x = self.conv_layers(input)
        x = x.view(x.size(0), -1)
        out = self.output_layer(x)

        return out



class Generator(nn.Module):
    '''Generator '''

    def __init__(self):
        super(Generator,self).__init__()

        gf_dim = params.gf_dim
        z_dim = params.z_dim
        c_dim = params.c_dim
        basic_h_size = params.height // 2 ** 4
        basic_w_size = params.width // 2 ** 4

        self.Linear = nn.Linear(z_dim, gf_dim*8*basic_h_size*basic_w_size)

        self.transposed_conv_layers = nn.Sequential(
            nn.BatchNorm2d(gf_dim*8),
            nn.ReLU(),
            nn.ConvTranspose2d(gf_dim*8, gf_dim*4, 4, 2, 1),
            nn.BatchNorm2d(gf_dim*4),
            nn.ReLU(),
            nn.ConvTranspose2d(gf_dim * 4, gf_dim * 2, 4, 2, 1),
            nn.BatchNorm2d(gf_dim*2),
            nn.ReLU(),
            nn.ConvTranspose2d(gf_dim * 2, gf_dim, 4, 2, 1),
            nn.BatchNorm2d(gf_dim),
            nn.ReLU(),
            nn.ConvTranspose2d(gf_dim, c_dim, 4, 2, 1),
            nn.Tanh()
        )


    def forward(self, input):
        basic_h_size = params.height // 2 ** 4
        basic_w_size = params.width // 2 ** 4

        x = self.Linear(input)
        x = x.view(-1, params.gf_dim*8, basic_h_size, basic_w_size)
        out = self.transposed_conv_layers(x)

        return out