'''

ENV :

python                    3.6.10               h9f7ef89_2
RTX 2080Ti driver :       451.67

Setup :
create new conda environment
$ conda install tensorflow-gpu=1.14

cudatoolkit               10.0.130                      0
cudnn                     7.6.5                cuda10.0_0
tensorboard               1.14.0           py36he3c9ec2_0
tensorflow                1.14.0          gpu_py36h305fd99_0
tensorflow-base           1.14.0          gpu_py36h55fc52a_0
tensorflow-estimator      1.14.0                     py_0
tensorflow-gpu            1.14.0               h0d30ee6_0

$ conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch

pytorch                   1.2.0           py3.6_cuda100_cudnn7_1
torchvision               0.4.0                py36_cu100

$ conda install matplotlib

matplotlib                3.2.2                         0
matplotlib-base           3.2.2            py36h64f37c6_0

$ conda install opencv

opencv                    3.3.1            py36h20b85fd_1

'''

import tensorflow as tf
from tensorflow import keras

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def reset_keras():
    sess = keras.backend.get_session()
    keras.backend.clear_session()
    sess.close()


def tf_test(name):
    # Basic Image classification test code

    print(f'This is , {name}')
    print(tf.__version__)

    # GPU memory growth
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    keras.backend.set_session(sess)

    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=10)

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

    print('\nTest accuracy:', test_acc)

    reset_keras()


def torch_test(name):
    print(f'This is , {name}')
    print(torch.__version__)

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # show images
    # imshow(torchvision.utils.make_grid(images))     # if you have already installed matplotlib
    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

    device = torch.device("cuda:0")
    net = Net()
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            # inputs, labels = data  # for cpu
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')


if __name__ == '__main__':
    tf_test('test code for tensorflow')
    print('\n\n=======================================\n\n')
    torch_test('test code for pytorch')
