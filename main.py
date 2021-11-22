import argparse
from torchvision import transforms

from utils import str2bool
from CONFIG import *
from dataload_utils import get_transforms, load_dataset, get_inference_data, ToTensor
from train import train_network
from inference import get_prediction_folder
from class_maps import CLASS_MAP
from pprint import pprint

parse = argparse.ArgumentParser(description='Parameters for Training')

parse.add_argument('-train', action='store', dest='train', help='Bool, yes/true/t/y/1/True if u want to train',
                   type=str2bool, nargs='?', const=True, default=False)

parse.add_argument('-predict_by_folder', action='store', dest='predict',
                   help='Bool, yes/true/t/y/1/True if u want to predict from a folder', type=str2bool,
                   nargs='?', const=True, default=False)

parse.add_argument('-data_dir_train', action='store', dest='train_dir',
                   help='Path to train directory, arrangement of sub-folders must be as per classes', type=str,
                   default=None)

parse.add_argument('-data_dir_test', action='store', dest='test_dir',
                   help='Path to test directory, arrangement of sub-folders must be as per classes', type=str,
                   default=None)

parse.add_argument('-data_dir_infer', action='store', dest='infer_dir',
                   help='Path to Inference directory', type=str, default=None)

parse.add_argument('-network', action='store', dest='network',
                   help='Name of the network to train or infer upon(default `CustomNetwork`),\nCustomNetwork\nOR\nresnet18',
                   type=str, default='CustomNetwork')

parse.add_argument('-input_channels', action='store', dest='input_channels',
                   help='Number of Input Channels for Images', type=int, default=1)

parse.add_argument('-epochs', action='store', dest='epochs',
                   help='Number of Epochs for training', type=int, default=5)

parse.add_argument('-batch_size', action='store', dest='batch_size',
                   help='Batch Size for training', type=int, default=100)

parse.add_argument('-lr', action='store', dest='learning_rate',
                   help='learning rate', type=float, default=0.001)

parse.add_argument('-model_save_path', action='store', dest='model_save_path',
                   help='Folder Path for saving trained models', type=str, default='models/')

parse.add_argument('-model_path_to_infer', action='store', dest='model_path_to_infer',
                   help='Path for loading saved model', type=str, default='./saved_fashion_mnist_model/Epoch_4.pt')

parse.add_argument('-class_map', action='store', dest='class_map',
                   help='Create a dict referring to all the classes in your dataset,\n check class_maps.py for reference',
                   type=str, default='FashionMNIST')

arguments = parse.parse_args()

if arguments.train:
    if not arguments.train_dir or not arguments.test_dir:
        raise Exception(f'Please provide Training and Testing data paths')

    network = arguments.network
    network_setting = NETWORK[network]
    input_size = network_setting["input_size"]
    input_channel = arguments.input_channels
    transform = get_transforms(image_size=input_size,
                               inp_channel=input_channel)

    train_path = arguments.train_dir
    test_path = arguments.test_dir
    batch_size = arguments.batch_size
    epochs = arguments.epochs
    lr = arguments.learning_rate
    model_save_folder = arguments.model_save_path

    dataloader_train, dataloader_test, train_size, test_size, class_names = load_dataset(train_path,
                                                                                         test_path,
                                                                                         batch_size=batch_size,
                                                                                         transform=transform)

    train_network(train_data=dataloader_train,
                  test_data=dataloader_test,
                  network=network,
                  epochs=epochs,
                  learning_rate=lr,
                  save_path=model_save_folder)

if arguments.predict:
    if not arguments.infer_dir:
        raise Exception(f'Please give directory of images to infer')

    folder_path = arguments.infer_dir

    network = arguments.network
    network_setting = NETWORK[network]
    input_size = network_setting["input_size"]
    input_channel = arguments.input_channels
    saved_model = arguments.model_path_to_infer
    class_map = CLASS_MAP[arguments.class_map]

    dataloader, _ = get_inference_data(folder_path, input_channel, input_size,
                                       transform=transforms.Compose([ToTensor()]))
    results = get_prediction_folder(dataloader,
                                    network,
                                    saved_model,
                                    class_map,
                                    input_channel)

    pprint(results)

# To Train: python main.py -train t -data_dir_train ./image_data/train -data_dir_test ./image_data/test
# To Infer: python main.py -predict t -data_dir_infer ./test_images/
