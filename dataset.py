from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
import json
import cv2


class DogBreedDataset(Dataset):
    def __init__(self, annotation_path):
        super(DogBreedDataset, self).__init__()
        with open(annotation_path, 'r') as f:
            self.annotation = list(json.load(f).items())

        self.transforms = Compose([
            ToTensor(),
            Resize((224, 224))
        ])

    def __getitem__(self, item):
        image_path = 'data/train/' + self.annotation[item][0]
        image = cv2.imread(image_path)
        image = self.transforms(image)

        label = self.annotation[item][1]

        return image, label

    def __len__(self):

        return len(self.annotation)


if __name__ == '__main__':
    dataset = DogBreedDataset(annotation_path='data/train.json')
    print(dataset[0][0].shape)
    pass
