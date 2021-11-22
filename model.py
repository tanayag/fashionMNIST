import torch
import torch.nn as nn
import torchvision


def get_fcn(num_nodes, len_classes):
    module = list()

    module.append(nn.Linear(num_nodes, 128))
    module.append(nn.BatchNorm1d(128))
    module.append(nn.ReLU())

    module.append(nn.Linear(128, 32))
    module.append(nn.BatchNorm1d(32))
    module.append(nn.ReLU())

    module.append(nn.Linear(32, len_classes))
    module.append(nn.Softmax())

    fcn = nn.Sequential(*module)
    return fcn


def resnet18(input_channel, len_classes):
    network = torchvision.models.resnet18(pretrained=True)
    num_nodes = network.fc.in_features
    del network.fc
    network.fc = get_fcn(num_nodes, len_classes)
    return network


class CustomNetwork(nn.Module):

    def __init__(self, inp_channel, output_features):
        super(CustomNetwork, self).__init__()

        self.inp_channel = inp_channel
        self.output_features = output_features

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=self.inp_channel, out_channels=16, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=64 * 4 * 4, out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=512, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=128, out_features=self.output_features),
            nn.Softmax()
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        # print(out.shape)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)

        return out


if __name__ == "__main__":
    print(CustomNetwork(1, 10))
