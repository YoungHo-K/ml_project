"""
Reference: https://www.kaggle.com/code/haqishen/train-efficientnet-b0-w-36-tiles-256-lb0-87/notebook
"""

import numpy as np


class ImageTilingProcessor:
    def __init__(self, size_of_tile, number_of_tiles):
        self.size_of_tile = size_of_tile
        self.number_of_tiles = number_of_tiles

    def process(self, image):
        image_tile_list = self._get_image_tiles(image)
        transformed_image = self._combine_image_tiles(image_tile_list)

        return transformed_image

    def _combine_image_tiles(self, image_tile_list):
        number_of_row_tiles = int(np.sqrt(self.number_of_tiles))

        image = np.zeros((self.size_of_tile * number_of_row_tiles, self.size_of_tile * number_of_row_tiles, 3))
        for height in range(0, number_of_row_tiles):
            for width in range(0, number_of_row_tiles):
                index = height * number_of_row_tiles + width

                if index < self.number_of_tiles:
                    current_image = image_tile_list[index]

                else:
                    current_image = np.ones((self.size_of_tile, self.size_of_tile, 3)).astype(np.uint8) * 255

                current_image = 255 - current_image

                height_index = height * self.size_of_tile
                width_index = width * self.size_of_tile
                image[height_index: height_index + self.size_of_tile, width_index: width_index + self.size_of_tile] = current_image

        image = image.astype(np.float32)
        image /= 255

        return image

    def _get_image_tiles(self, image):
        height, width, channel = image.shape

        padding_height = (self.size_of_tile - height % self.size_of_tile) % self.size_of_tile
        padding_width = (self.size_of_tile - width % self.size_of_tile) % self.size_of_tile
        padding_images = np.pad(array=image,
                                pad_width=[[padding_height // 2, padding_height - padding_height // 2],
                                           [padding_width // 2, padding_width - padding_width // 2],
                                           [0, 0]],
                                constant_values=255)

        reshaped_images = padding_images.reshape(padding_images.shape[0] // self.size_of_tile, self.size_of_tile,
                                                 padding_images.shape[1] // self.size_of_tile, self.size_of_tile, 3)
        reshaped_images = reshaped_images.transpose(0, 2, 1, 3, 4).reshape(-1, self.size_of_tile, self.size_of_tile, 3)
        if len(reshaped_images) < self.number_of_tiles:
            reshaped_images = np.pad(array=reshaped_images,
                                     pad_width=[[0, self.number_of_tiles - len(reshaped_images)],
                                                [0, 0],
                                                [0, 0],
                                                [0, 0]],
                                     constant_values=255)

        variance_of_pixel_values = reshaped_images.mean(-1).reshape(reshaped_images.shape[0], -1).var(-1)

        max_image_filter = variance_of_pixel_values <= np.percentile(variance_of_pixel_values, 98)
        min_image_filter = np.percentile(variance_of_pixel_values, 2) <= variance_of_pixel_values

        reshaped_images = reshaped_images[np.where(max_image_filter & min_image_filter)]

        indexes = np.argsort(reshaped_images.reshape(reshaped_images.shape[0], -1).sum(-1))[: self.number_of_tiles]
        reshaped_images = reshaped_images[indexes]

        return reshaped_images


if __name__ == "__main__":
    import os
    import cv2
    import pandas as pd

    ROOT_DIR_PATH = "/Users/youngho/Documents/GitHub/ml_project/dacon/resource"
    FILE_NAME = "train.csv"
    NUMBER_OF_TILES = 25
    SIZE_OF_TILE = 256

    image_tiling_processor = ImageTilingProcessor(size_of_tile=SIZE_OF_TILE, number_of_tiles=NUMBER_OF_TILES)

    data_frame = pd.read_csv(os.path.join(ROOT_DIR_PATH, FILE_NAME))
    for index in range(0, len(data_frame)):
        data_frame["img_path"].iloc[index] = os.path.join(ROOT_DIR_PATH, data_frame["img_path"].iloc[index][2:])

    for index in range(0, len(data_frame)):
        image_path = data_frame["img_path"].iloc[index]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image_tiling_processor.process(image)

        cv2.imshow("test", image)
        cv2.waitKey(0)
