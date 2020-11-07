from .voc_data_generator import  VOCDataGenerator
import torchvision.transforms as transforms
import torch


class VOCDataBatchGenerator(VOCDataGenerator):
    def __init__(self, to_fit, images_dir, label_txt, image_shape=(448, 448, 3), grid_size=7, max_bb_count=2, num_classes=20, batch_size=1, dtype=torch.float):
        VOCDataGenerator.__init__(self, to_fit, images_dir, label_txt, image_shape, grid_size, max_bb_count, num_classes)
        self.batch_size = batch_size
        self.dtype = dtype

    def __getitem__(self, idx):
        batch = []
        for i in range(idx * self.batch_size, (idx + 1) * self.batch_size):
            batch.append(VOCDataGenerator.__getitem__(self, i))
        return transforms.ToTensor(batch).type(self.dtype)