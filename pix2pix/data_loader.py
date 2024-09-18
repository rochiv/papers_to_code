import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class CityscapesDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [
            f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name)

        # Split the image into left (real) and right (outlined) parts
        width, height = image.size
        real_image = image.crop((0, 0, width // 2, height))
        outlined_image = image.crop((width // 2, 0, width, height))

        if self.transform:
            real_image = self.transform(real_image)
            outlined_image = self.transform(outlined_image)

        return {"real_image": real_image, "outlined_image": outlined_image}


# Example usage
if __name__ == "__main__":
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    dataset = CityscapesDataset(
        root_dir="/media/rohit/mace/paper_to_code/pix2pix/cityscapes/train",
        transform=transform,
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    for i, sample in enumerate(dataloader):
        print(sample["real_image"].shape, sample["outlined_image"].shape)
        if i == 1:
            break
