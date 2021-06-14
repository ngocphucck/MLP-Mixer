import pandas as pd
import matplotlib.pyplot as plt
import os
import random
from PIL import Image
import json
from sklearn.model_selection import train_test_split


def explore_data(label_path='data/labels.csv',
                 data_path='data/train'):
    labels = pd.read_csv(label_path)
    labels = dict(zip(labels['id'], labels['breed']))
    image_names = os.listdir(data_path)

    image_name = image_names[random.randint(0, len(labels))]
    image_path = os.path.join(data_path, image_name)

    image = Image.open(image_path)
    plt.imshow(image)
    print(f'Image id: {image_name}')
    print(f'Label: {labels[image_name[:-4]]}')
    print(f'Image size: {image.size}')

    plt.show()


def encode(label_path='data/labels.csv'):
    labels = pd.read_csv(label_path)
    labels = set(labels['breed'])
    labels = sorted(list(labels))

    return {category: id for id, category in enumerate(labels)}


def split_train_val(image_path='data/train',
                    label_path='data/labels.csv',
                    save_path='data'):

    labels = encode(label_path)
    annotation = pd.read_csv(label_path)
    annotation = dict(zip(annotation['id'], annotation['breed']))
    image_paths = list(os.listdir(image_path))

    train_images, val_images = train_test_split(image_paths, test_size=0.2, shuffle=True, random_state=55)
    train_data = {image: labels[annotation[image[:-4]]] for image in train_images}
    val_data = {image: labels[annotation[image[:-4]]] for image in val_images}

    with open(save_path + '/train.json', 'w') as f:
        json.dump(train_data, f)

    with open(save_path + '/val.json', 'w') as f:
        json.dump(val_data, f)


if __name__ == '__main__':
    pass
