'''Dataset setting and data loader for MNIST.'''

import torch
from torchvision import datasets, transforms
from config import params

def get_mnist(train=True):
    """Get MNIST dataset loader."""
    # image pre-processing
    pre_process = transforms.Compose([transforms.Resize([params.height, params.width]),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                      ])

    # dataset and data loader
    mnist_dataset = datasets.MNIST(root=params.dataset_root,
                                   train=train,
                                   transform=pre_process,
                                   download=True)

    mnist_data_loader = torch.utils.data.DataLoader(
        dataset=mnist_dataset,
        batch_size=params.batch_size,
        shuffle=True)

    return mnist_data_loader
