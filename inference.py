from model import *
from dataload_utils import *
from CONFIG import *

from torchvision import datasets, transforms
import torch

from skimage import io, color
from skimage.transform import resize

from class_maps import CLASS_MAP


def get_prediction_folder(dataloader, network, model_path, class_map, input_channel):
    result_lst = list()
    output_features = len(class_map)
    network = eval(network)(input_channel, output_features)
    network.load_state_dict(torch.load(model_path))
    network.eval()

    results_dict = dict()
    for image, name_image in dataloader:
        out = network(image.float())
        preds = torch.max(out, 1)[1]
        preds = class_map[int(preds)]
        results_dict[name_image] = preds

    return results_dict


def get_prediction_image(img_path, network, model_path, input_size, input_channel, class_map):
    output_features = len(class_map)
    network = eval(network)(input_channel, output_features)
    network.load_state_dict(torch.load(model_path))
    network.eval()

    img = color.rgb2gray(io.imread(img_path))
    img = resize(img, (input_size, input_size))
    img = img.reshape((1, input_channel, input_size, input_size))
    img = torch.from_numpy(img)

    out = network(img.float())
    pred = torch.max(out, 1)[1]
    pred = class_map[int(pred)]

    return pred

if __name__ == "__main__":
    from class_maps import CLASS_MAP

    input_channel = 1
    net = "CustomNetwork"
    network_setting = NETWORK[net]
    input_size = network_setting["input_size"]

    infer_dataloader, len_test_cases = get_inference_data("/home/tanay/interviews/vector_ai/fashion_mnist/test_images",
                                                          input_channel, input_size, transforms.Compose([ToTensor()]))
    print(get_prediction_folder(infer_dataloader, net, "/home/tanay/interviews/vector_ai/fashion_mnist/models/Epoch_4.pt",
          CLASS_MAP["FashionMNIST"], input_channel))

    print(get_prediction_image('./test_images/319.jpeg',
                               net,
                               './models/Epoch_4.pt',
                               input_size,
                               input_channel,
                               CLASS_MAP["FashionMNIST"]))
