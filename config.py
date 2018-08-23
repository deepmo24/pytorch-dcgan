import argparse

parser = argparse.ArgumentParser()

# environment parameters
parser.add_argument('--gpu_id', default='0', help='if using multi-gpu: 0,1 | 1,2 | 0,1,2')
parser.add_argument('--result_root', default='results/', help='result root')
parser.add_argument('--model_root', default='checkpoints/', help='save model, subdir of result root')
parser.add_argument('--samples_root', default='samples/', help='save generated samples during training, subdir of result root')
parser.add_argument('--generator_restored', default=None, help='restored G model path')
parser.add_argument('--discriminator_restored', default=None, help='restored D model path')
parser.add_argument('--mode', default='train', help='train | test')
parser.add_argument('--manual_seed', default=None, help='manually set random seed')

# dataset parameters
parser.add_argument('--dataset', default='mnist', help='dataset name, mnist | celeba | (your dataset)')
parser.add_argument('--dataset_root', default='data/mnist', help='dataset root')
parser.add_argument('--c_dim', type=int, default=1, help='channel of input images')
parser.add_argument('--height', type=int, default=32, help='heigt of resized input images, input images will be resized by this size. dcgan model requires height to be a multiple of 16(2**4)')
parser.add_argument('--width', type=int, default=32, help='width of resized input images, input images will be resized by this size. dcgan model requires width to be a multiple of 16(2**4)')
parser.add_argument('--z_dim', type=int, default=100, help='Dimension of noise.')

# model parameters
parser.add_argument('--df_dim', type=int, default=64, help="Dimension of D's filters in first conv layer.")
parser.add_argument('--gf_dim', type=int, default=64, help="Dimension of G's filters in first conv layer.")

# training parameters
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--max_epoch', type=int, default=25, help='how many epochs to train')
parser.add_argument('--log_step', type=int, default=5, help='print loss information')
parser.add_argument('--save_epoch', type=int, default=2, help='save models every save_epoch')
parser.add_argument('--d_lr', type=float, default=0.0002, help='learning rate of discriminator')
parser.add_argument('--g_lr', type=float, default=0.0002, help='learning rate of generator')
parser.add_argument('--beta_a', type=float, default=0.5, help='beta a of Adam optimizer')
parser.add_argument('--beta_b', type=float, default=0.999, help='beta b of Adam optimizer')

# testing parameters
parser.add_argument('--num_images', type=int, default=1000, help=' number of images generated')
parser.add_argument('--save_root', default='generated_images/', help='save generated images during testing, subdir of result root')




params = parser.parse_args()

# set gpu id
gpu_ids = params.gpu_id.split(',')
gpu_ids = [int(i) for i in gpu_ids]
params.gpu_id = gpu_ids

# combine dir
params.model_root = params.result_root + params.model_root
params.samples_root = params.result_root + params.samples_root
params.save_root = params.result_root + params.save_root

