import os
import numpy as np
import torch
from torch import nn
import math
import torchvision
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch.utils import data
from PIL import Image


#___________________________________________
#ResNet

#For ResNet 18, 34, use two 3 x 3 convolution
class BasicBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride = 1):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_planes, planes, kernel_size = 3, 
            stride = stride, padding = 1, bias = False)
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size =3,
            stride = 1, padding = 1, bias = False)
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.shortcut = torch.nn.Sequential()

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_planes, self.expansion * planes,
                    kernel_size = 1, stride = stride, bias = False),
                torch.nn.BatchNorm2d(self.expansion*planes)
                )

    def forward(self, x):
        out = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        out = torch.nn.functional.relu(self.bn2(self.conv2(out)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.nn.functional.relu(out)
        return out

class ResNet(torch.nn.Module):
    def __init__(self, block, num_blocks, input_channels = 1, num_classes = 10):
        super().__init__()
        self.in_planes = 64

        self.conv1 = torch.nn.Conv2d(input_channels, 64, kernel_size = 3,
            stride = 1, padding = 1, bias = False)
        self.bn1 = torch.nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride = 1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride = 2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride = 2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride = 2)
        self.linear = torch.nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        out = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = torch.nn.functional.avg_pool2d(out, 4)

        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

#___________________________________________
#VGG 11

class VGG(nn.Module):

    def __init__(self, features, num_classes=10, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, input_channels = 1, batch_norm=False):
    layers = []
    in_channels = input_channels
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class AlexNet(torch.nn.Module):   
    def  __init__(self, input_channels = 1, output_size = 10):
        super().__init__()
        self.conv1 = torch.nn.Sequential(   #input_size = 227*227*1
            torch.nn.Conv2d(input_channels, 96, 11, 4, 0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2) #output_size = 27*27*96
        )
        self.conv2 = torch.nn.Sequential(   #input_size = 27*27*96
            torch.nn.Conv2d(96, 256, 5, 1, 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3, 2)    #output_size = 13*13*256
        )
        self.conv3 = torch.nn.Sequential(   #input_size = 13*13*256
            torch.nn.Conv2d(256, 384, 3, 1, 1), 
            torch.nn.ReLU(),    #output_size = 13*13*384
        )
        self.conv4 = torch.nn.Sequential(   #input_size = 13*13*384
            torch.nn.Conv2d(384, 384, 3, 1, 1),
            torch.nn.ReLU(),    #output_size = 13*13*384
        )
        self.conv5 = torch.nn.Sequential(   #input_size = 13*13*384
            torch.nn.Conv2d(384, 256, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3, 2)    #output_size = 6*6*256
        )
 
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(9216, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, output_size)
        )
 
    def forward(self, x): 
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        conv5_out = self.conv5(x)
        x = conv5_out.view(conv5_out.size(0), -1)
        out = self.dense(x)
        return out

class FlameSet(data.Dataset):
    def __init__(self,root, image_size = 28):
        self.imgs_list = os.listdir(root)
        self.imgs = [os.path.join(root,k) for k in self.imgs_list]
        self.labels = [k.split('-')[0] for k in self.imgs_list]
        self.transforms = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
            ])

    def __getitem__(self, index):
        img_path = self.imgs[index]
        pil_img = Image.open(img_path)
        pil_img = pil_img.convert('L')
        if self.transforms:
            data = self.transforms(pil_img)
        else:
            pil_img = np.asarray(pil_img)
            data = torch.from_numpy(pil_img)
        return data

    def __len__(self):
        return len(self.imgs)

    def get_image_list(self):
        return self.imgs_list

    def get_image_label(self):
        return self.labels

def resize_image(input_size = 28):
    transform = transforms.Compose([
    transforms.Resize(input_size), 
    transforms.ToTensor()
    ])
    return transform

def view_datasets(image_loader, objective_list):
    objective_list = np.array(objective_list)
    images, labels = next(iter(image_loader))
    img = torchvision.utils.make_grid(images)
    img = img.numpy().transpose(1, 2, 0)
    print(objective_list[labels.tolist()])
    # plt.axis('off')
    # plt.imshow(img)
    return (images, objective_list[labels.tolist()])

def create_network(network):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    res = network
    net = res.to(device)
    return net

def train_model(net, train_loader, LR, epochs = 1, number_of_images = None):
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        net.parameters(),
        lr=LR
    )
    for epoch in range(epochs):
        sum_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            if torch.cuda.is_available():
                inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
            else:
                inputs, labels = Variable(inputs).cpu(), Variable(labels).cpu()
            optimizer.zero_grad()  #Make gradient to zero
            outputs = net(inputs)  #Forward calculation 
            loss = loss_function(outputs, labels)  #Get loss function
            loss.backward()  #back propogation
            optimizer.step()  #Update parameter.
            # print(loss)
            sum_loss += loss.item()
            if i % 100 == 99:
                print('[%d,%d] loss:%.03f' %
                      (epoch + 1, i + 1, sum_loss / 100))
                sum_loss = 0.0
            if number_of_images is None:
                pass
            else:
                if i * train_loader.batch_size >= number_of_images:
                    print ("Current Loss {}".format(sum_loss))
                    break
    return net

def test_model(net, test_loader, number_of_images = None):
    net.eval()  #Convert to test model
    correct = 0
    total = 0
    for i, data_test in enumerate(test_loader):
        print (i)
        images, labels = data_test
        if torch.cuda.is_available():
            images, labels = Variable(images).cuda(), Variable(labels).cuda()
        else:
            images, labels = Variable(images).cpu(), Variable(labels).cpu()
        output_test = net(images)
        _, predicted = torch.max(output_test, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        if number_of_images is None:
            pass
        else:
            if i * test_loader.batch_size >= number_of_images:
                print("correct1: ", correct.item())
                print("Test acc: {0}".format(correct.item() /
                                     ((i+1) * test_loader.batch_size)))
                break
    if number_of_images is None:
        print("correct1: ", correct.item())
        print("Test acc: {0}".format(correct.item() /
                                     len(test_loader)))

def predict_image(net, input_image, objective_list, num_of_prediction = 1):
    net.eval()
    objective_list = np.array(objective_list)
    if isinstance(input_image, torch.utils.data.DataLoader):
        images, labels = view_datasets(input_image, objective_list)
        if torch.cuda.is_available():
            net = net.cuda()
            images = Variable(images).cuda()
        else:
            images = Variable(images).cpu()
    else:
        input_image = input_image.unsqueeze(-3)
        img = torchvision.utils.make_grid(input_image)
        img = img.numpy().transpose(1, 2, 0)
        plt.imshow(img)
        if torch.cuda.is_available():
            net = net.cuda()
            images = Variable(input_image).cuda()
        else:
            images = Variable(input_image).cpu()
    print (images)
    output_test = net(images)
    _, predicted = torch.max(output_test, 1)

    print ("The predict result is {}".format(objective_list[predicted.tolist()]))



