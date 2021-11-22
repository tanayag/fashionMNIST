import os

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import json
import argparse


def csv_to_image(data_csv_path, folder, class_map):
    """
    function converts csv to labeled images in folder
    :param data_csv_path:
    :param folder:
    :param class_map:
    :return:
    """
    data_csv = pd.read_csv(data_csv_path)
    if not os.path.exists(folder):
        os.mkdir(folder)

    img_count_naming = dict()
    for i in tqdm(range(len(data_csv))):
        if not os.path.exists(os.path.join(folder, str(class_map[int(data_csv.iloc[i]['label'])]))):
            os.mkdir(os.path.join(folder, str(class_map[int(data_csv.iloc[i]['label'])])))
            img_count_naming[str(data_csv.iloc[i]['label'])] = 1

        img = np.array(data_csv.iloc[i][1:])
        img = img.reshape(28, 28)
        img = Image.fromarray(np.uint8(img))
        img.save(os.path.join(folder, str(class_map[int(data_csv.iloc[i]['label'])]),
                              str(img_count_naming[str(data_csv.iloc[i]['label'])]) + '.jpeg'))
        img_count_naming[str(data_csv.iloc[i]['label'])] += 1


def load_json(file_path):
    with open(file_path) as f:
        dct = json.load(f)
    return dct


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def inverse_dict(dct):
    """ fucntion written for fashionMNIST, since the csv follow
    inverse lexicographical order"""
    l = len(dct)
    ele = []
    for i in range(l):
        ele.append(dct[i])
    ele = ele[::-1]
    inverse_dct = dict()
    for i in range(l):
        inverse_dct[i] = ele[i]

    return inverse_dct

if __name__ == "__main__":
    from class_maps import CLASS_MAP

    csv_to_image("./data/fashion-mnist_test.csv", 'image_data/test', inverse_dict(CLASS_MAP["FashionMNIST"]))

    print(inverse_dict(CLASS_MAP["FashionMNIST"]))
