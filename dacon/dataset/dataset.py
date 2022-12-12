import cv2
import torch

from torch.utils.data import Dataset

from dacon.dataset.image_tiling_processor import ImageTilingProcessor


class MultiModalDataset(Dataset):
    def __init__(self, medical_data_frame, labels=None, transform=None, size_of_tile=36, number_of_tiles=256):
        self.medical_data_frame = medical_data_frame
        self.labels = labels
        self.transfrom = transform
        self.image_tiling_processor = ImageTilingProcessor(size_of_tile, number_of_tiles)

    def __getitem__(self, index):
        image_path = self.medical_data_frame["img_path"].iloc[index]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.image_tiling_processor.process(image)

        if self.transfrom is not None:
            image = self.transfrom(image=image)["image"]

        if self.labels is not None:
            tabular = torch.Tensor(self.medical_data_frame.drop(columns=["ID", "img_path", "mask_path", "수술연월일"]).iloc[index])
            label = self.labels[index]

            return image, tabular, label

        tabular = torch.Tensor(self.medical_data_frame.drop(columns=["ID", "img_path", "수술연월일"]).iloc[index])

        return image, tabular

    def __len__(self):
        return len(self.medical_data_frame)