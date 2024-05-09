import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import glob


class Cityscapes(Dataset):
    def __init__(self, root: str = '.', transform=None, direction: str = 'B2A'):
        """
        Initialize the Cityscapes dataset class for loading images.
        Parameters:
        root (str): Root directory where the dataset is located.
        transform (callable, optional): A function/transform that takes in two PIL images and returns transformed versions.
        direction (str): Specifies the image conversion direction, 'A2B' or 'B2A'.
        """
        self.root = root
        self.files = sorted(glob.glob(f"{root}/cityscapes/*.jpg"))
        self.transform = transform
        self.direction = direction

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        Retrieve a dataset item by index.
        Parameters:
        idx (int): The index of the item.
        Returns:
        tuple: A tuple containing the transformed images.
        """
        img = Image.open(self.files[idx]).convert('RGB')
        W, H = img.size
        cW = W // 2
        imgA = img.crop((0, 0, cW, H))
        imgB = img.crop((cW, 0, W, H))

        if self.transform:
            imgA, imgB = self.transform(imgA), self.transform(imgB)

        if self.direction == 'A2B':
            return imgA, imgB
        else:
            return imgB, imgA

    def divide_into_sets(self, train_frac=0.7, val_frac=0.2):
        total_size = len(self)
        train_size = int(total_size * train_frac)
        val_size = int(total_size * val_frac)
        test_size = total_size - train_size - val_size

        train_set, val_set, test_set = random_split(self, [train_size, val_size, test_size])
        return train_set, val_set, test_set
    

    
