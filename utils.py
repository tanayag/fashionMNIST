import os

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import json


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


if __name__ == "__main__":
    from fashionMNIST_label import CLASS_MAP

    csv_to_image("./data/fashion-mnist_test.csv", 'image_data/test', CLASS_MAP)
