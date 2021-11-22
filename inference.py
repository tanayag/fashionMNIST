from model import *
from dataload_utils import *
from CONFIG import *

from torchvision import datasets, transforms
import torch

from skimage import io, color
from skimage.transform import resize


def get_prediction_folder(dataloader, network, model_path, class_map, input_channel):
    result_lst = list()
    output_features = len(class_map)
    network = eval(network)(input_channel, output_features)
    network.load_state_dict(torch.load(model_path))
    network.eval()
    for image in dataloader:
        out = network(image.float())
        print(out)
        preds = torch.max(out, 1)[1]
        preds = class_map[int(preds)]
        result_lst.append(preds)

    return result_lst


def get_prediction_image(image, network, model_path, class_map, input_channel):
    output_features = len(class_map)
    network = eval(network)(input_channel, output_features)
    network.load_state_dict(torch.load(model_path))
    network.eval()

    img = color.rgb2gray(io.imread(img_path))
    img = torch.from_numpy(img)

if __name__ == "__main__":
    from fashionMNIST_label import CLASS_MAP

    input_channel = 1
    net = "CustomNetwork"
    network_setting = NETWORK[net]
    input_size = network_setting["input_size"]

    infer_dataloader, len_test_cases = get_inference_data("/home/tanay/interviews/vector_ai/fashion_mnist/test_images",
                                                          input_channel, input_size, transforms.Compose([ToTensor()]))
    print(get_prediction(infer_dataloader, net, "/home/tanay/interviews/vector_ai/fashion_mnist/models/Epoch_4.pt",
                         CLASS_MAP, input_channel))
