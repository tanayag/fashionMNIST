from torch.autograd import Variable

from model import *
from dataload_utils import *
from CONFIG import *


def train_network(train_data,
                  test_data,
                  network,
                  epochs,
                  learning_rate,
                  save_path):
    """

    :param train_data:
    :param test_data:
    :param network:
    :param epochs:
    :param learning_rate:
    :param save_path:
    :return:
    """
    criterion = nn.CrossEntropyLoss()
    network = eval(network)(inp_channel=1, output_features=10)
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

    running_loss_train = list()
    running_accuracy_train = list()
    running_loss_test = list()
    running_accuracy_test = list()

    for epoch in range(epochs):
        for images, labels in train_data:
            network.train()
            images = Variable(images)
            labels = Variable(labels)

            predicted_labels = network(images)

            loss = criterion(predicted_labels, labels)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

        count_test_image = 0
        correct_preds = 0

        for images_test, labels_test in test_data:
            images_test, labels_test = Variable(images_test), Variable(labels_test)
            network.eval()
            predicted_labels_test = network(images_test)
            predictions = torch.max(predicted_labels_test, 1)[1]

            correct_preds += sum([pred == label for pred, label in zip(predictions, labels_test)])
            count_test_image += len(labels_test)

        test_accuracy = correct_preds * 100 / count_test_image
        print(f'Test Accuracy : {test_accuracy}')

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        torch.save(network.state_dict(), os.path.join(save_path, f"Epoch_{epoch}.pt"))


if __name__ == "__main__":
    train_path = '/home/tanay/interviews/vector_ai/fashion_mnist/image_data/train'
    test_path = '/home/tanay/interviews/vector_ai/fashion_mnist/image_data/test'

    input_channel = 1
    EPOCHS = 5
    LR = 0.001
    BATCH_SIZE = 100

    net = "CustomNetwork"
    network_setting = NETWORK[net]
    input_size = network_setting["input_size"]

    transform = get_transforms(image_size=input_size,
                               inp_channel=input_channel)

    dataloader_train, dataloader_test, train_size, test_size, class_names = load_dataset(train_path,
                                                                                         test_path,
                                                                                         batch_size=BATCH_SIZE,
                                                                                         transform=transform)

    train_network(train_data=dataloader_train,
                  test_data=dataloader_test,
                  network=net,
                  epochs=EPOCHS,
                  learning_rate=LR,
                  save_path="models/")
