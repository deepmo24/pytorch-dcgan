'''Dataset setting and data loader for celeba.'''

import os
import torch.utils.data
from torchvision import transforms, datasets
from config import params



def get_celeba(Train=True):
    '''Get celeba dataset's dataloader'''

    pre_process = transforms.Compose([
        transforms.Resize([params.height, params.width]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    celeba_dataset = datasets.ImageFolder(root=params.dataset_root,
                                          transform=pre_process)

    celeba_dataloader = torch.utils.data.DataLoader(
        dataset=celeba_dataset,
        batch_size= params.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4
    )

    return celeba_dataloader