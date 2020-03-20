import glob
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

gopro_default_test_path = ''


class TestDataset(Dataset):
    def __init__(self, dataset_path=gopro_default_test_path):
        self.tensor_setup = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        self.files_blurred = sorted(glob.glob(os.path.join(dataset_path, "*")))

    def __getitem__(self, index):
        filePath_blurred = self.files_blurred[index % len(self.files_blurred)]
        data = np.array(Image.open(filePath_blurred), 'f') / 255.

        return self.tensor_setup(data)

    def __len__(self):
        return len(self.files_blurred)
