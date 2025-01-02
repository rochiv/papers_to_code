import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib  # For loading NIfTI files

class MedicalImageDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_filenames = sorted(os.listdir(image_dir))
        self.label_filenames = sorted(os.listdir(label_dir))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        label_path = os.path.join(self.label_dir, self.label_filenames[idx])

        # Load the image and label
        image = nib.load(image_path).get_fdata()
        label = nib.load(label_path).get_fdata()

        # Convert to torch tensors
        image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        # Apply transformations if any
        if self.transform:
            image, label = self.transform(image, label)

        return image, label

def get_dataloader(image_dir, label_dir, batch_size=1, shuffle=True, num_workers=0, transform=None):
    dataset = MedicalImageDataset(image_dir, label_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

# Example usage
# image_dir = 'path/to/images'
# label_dir = 'path/to/labels'
# dataloader = get_dataloader(image_dir, label_dir, batch_size=2)
# for images, labels in dataloader:
#     print(images.shape, labels.shape)
