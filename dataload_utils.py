import os

import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

from skimage import io, color
from skimage.transform import resize


def load_dataset(train_dir, test_dir, batch_size, transform=None):
    train_data = datasets.ImageFolder(train_dir, transform)
    test_data = datasets.ImageFolder(test_dir, transform)

    dataloader_train = DataLoader(train_data, batch_size=batch_size,
                                  shuffle=True, num_workers=4)
    dataloader_test = DataLoader(test_data, batch_size=batch_size,
                                 shuffle=True, num_workers=4)

    train_size = len(train_data)
    test_size = len(test_data)
    class_names = train_data.classes

    return dataloader_train, dataloader_test, train_size, test_size, class_names


def get_transforms(image_size, inp_channel):
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=inp_channel),
                                    transforms.Resize(image_size),
                                    transforms.ToTensor()])
    return transform


class LoadInferenceData(Dataset):
    def __init__(self, image_folder_path, input_channels, input_size, transform=None):
        """

        :param image_folder_path:
        :param input_channels:
        """
        img_paths = os.listdir(image_folder_path)
        complete_imgs_path = [os.path.join(image_folder_path, img_path) for img_path in img_paths]
        self.complete_imgs_path = complete_imgs_path
        self.input_channels = input_channels
        self.transform = transform
        self.input_size = input_size

    def __len__(self):
        return len(self.complete_imgs_path)

    def __getitem__(self, idx):
        img_path = self.complete_imgs_path[idx]
        if self.input_channels == 1:
            img = color.rgb2gray(io.imread(img_path))
        else:
            img = io.imread(img_path)
        img = resize(img, (self.input_size, self.input_size))
        img = img.reshape(self.input_channels, *img.shape)
        if self.transform:
            img = self.transform(img)
        return img


class ToTensor(object):
    def __call__(self, image):
        return torch.from_numpy(image)


def get_inference_data(image_folder_path, input_channels, input_size, transform=None):
    """

    :param image_folder_path:
    :param input_channels:
    :param input_size:
    :param transform:
    :return:
    """
    image_loader = LoadInferenceData(image_folder_path, input_channels, input_size, transform=transform)
    dataloader = DataLoader(image_loader, batch_size=1)
    return dataloader, len(image_loader)


if __name__ == "__main__":
    x, lg = get_inference_data("/home/tanay/interviews/vector_ai/fashion_mnist/test_images",
                               1, 28, transform=transforms.Compose([ToTensor()]))
    for i in x:
        print(i.shape)
