from CONFIG import NETWORK
from class_maps import CLASS_MAP
from model import *

input_channel = 1
network = "CustomNetwork"
network_setting = NETWORK[network]
input_size = network_setting["input_size"]
model_path = "./models/Epoch_4.pt"
class_map = CLASS_MAP["FashionMNIST"]

def get_prediction_image(img_arr, batch_size):
    output_features = len(class_map)
    network = eval(network)(input_channel, output_features)
    network.load_state_dict(torch.load(model_path))
    network.eval()

    img = resize(img_arr, (input_size, input_size))
    img = img.reshape((batch_size, input_channel, input_size, input_size))
    img = torch.from_numpy(img)

    out = network(img.float())
    pred = torch.max(out, 1)[1]
    pred = class_map[int(pred)]

    return pred