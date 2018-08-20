'''parameters'''


# environment parameters
gpu_id = [2] #must list now
model_root = 'result/checkpoints/'
samples_root = 'result/samples/'
generator_restored = None # model path
discriminator_resotred = None # model path
mode = 'train' # train or test mode

# dataset parameters
dataset_root = '/home/dataset/celeba'
c_dim = 3   # channel
height = 64 # dcgan model requires height to be a multiple of 16(2**4)
width = 64  # dcgan model requires width to be a multiple of 16(2**4)
z_dim = 100 # Dimension of noise.


# model parameters
df_dim = 64 # Dimension of D's filters in first conv layer.
gf_dim = 64 # Dimension of G's filters in first conv layer.


# training parameters
batch_size = 128
max_epoch = 25
log_step = 5
save_epoch = 2
d_lr = 0.0002
g_lr = 0.0002
betas = (0.5,0.999)

# testing parameters
num_images = 1000 # number of images generated
save_root = 'result/generated_images/'