from torch.utils.data import Dataset
import json
from PIL import Image
import torch


from utils.augment_utils import RandAugment, Resize, ToTensor, Normalize


class DogBreedDataset(Dataset):
    def __init__(self, annotation_path, width=256, height=256, num_aug=10, magnitude=15):
        super(DogBreedDataset, self).__init__()
        with open(annotation_path, 'r') as f:
            self.annotation = list(json.load(f).items())
        self.width = width
        self.height = height
        self.num_aug = num_aug
        self.magnitude = magnitude

    def _transform(self, image):
        image = RandAugment(self.num_aug, self.magnitude)(image)
        image = Resize(self.width, self.height)(image)
        image = ToTensor()(image)
        image = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(image)

        return image

    def __getitem__(self, item):
        image_path = 'data/train/' + self.annotation[item][0]
        image = Image.open(image_path)
        image = self._transform(image)

        label = self.annotation[item][1]
        label = torch.tensor(label, dtype=torch.long)

        return image, label

    def __len__(self):

        return len(self.annotation)


if __name__ == '__main__':
    dataset = DogBreedDataset(annotation_path='data/train.json')
    print(dataset[0][0])
    pass
