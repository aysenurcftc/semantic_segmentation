import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import cv2

from config import DATA_DIR


class Dataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask = mask / 255.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask


class PetDataset(Dataset):
    def __init__(self, root=DATA_DIR , is_train=True, transform=None):

        self.transform = transform
        self.classes = ['background', 'animal']
        self.root = root

        # Select the appropriate annotation file based on whether it's training or testing
        if is_train:
            annotations = os.path.join(root, "annotations", "trainval.txt")
        else:
            annotations = os.path.join(root, "annotations", "test.txt")

        # Read the annotation file and extract image names
        with open(annotations, 'r') as file:
            self.img_names = [image.split(' ')[0] for image in file.readlines()]

    def __len__(self):

        return len(self.img_names)

    def __getitem__(self, item):

        img_name = self.img_names[item]
        img_path = os.path.normpath(os.path.join(self.root, "images", img_name) + ".jpg")
        mask_path = os.path.normpath(os.path.join(self.root, "annotations", "trimaps", img_name) + ".png")

        # Read the image and convert it from BGR to RGB
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Read the mask and adjust its values
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # 0.299 ∙ Red + 0.587 ∙ Green + 0.114 ∙ Blue.
        mask[mask == 2] = 0  # Set background class to 0
        mask[mask == 3] = 1  # Set animal class to 1

        # Apply transformations if provided
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        return image, mask





