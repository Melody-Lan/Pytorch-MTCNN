import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from  PIL import  Image
import torch.nn.functional as F
class PNet(nn.Module):

    def __init__(self):
        super(PNet, self).__init__()

        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=3, stride=1),  # conv1
            nn.PReLU(),  # PReLU1
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool1
            nn.Conv2d(10, 16, kernel_size=3, stride=1),  # conv2
            nn.PReLU(),  # PReLU2
            nn.Conv2d(16, 32, kernel_size=3, stride=1),  # conv3
            nn.PReLU()  # PReLU3
        )

        self.conv4_1 = nn.Conv2d(32, 1, kernel_size=1, stride=1)
        self.conv4_2 = nn.Conv2d(32, 4, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.pre_layer(x)
        cond = torch.sigmoid(self.conv4_1(x))
        offset = self.conv4_2(x)
        return cond, offset

class Rnet(nn.Module):
    def __init__(self):
        super(Rnet, self).__init__()

        self.pre_layers = nn.Sequential(

            nn.Conv2d(3, 28, kernel_size=3, stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.PReLU(),
            nn.Conv2d(28, 48, kernel_size=3, stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(48, 64, kernel_size=2, stride=1),
            nn.PReLU(),

        )
        self.conv4 = nn.Linear(64 * 2 * 2, 128)  # conv4
        self.prelu4 = nn.PReLU()  # prelu4
        # detection
        self.conv5_1 = nn.Linear(128, 1)
        # bounding box regression
        self.conv5_2 = nn.Linear(128, 4)

    def forward(self, x):
        # backend
        x = self.pre_layer(x)
        x = x.view(x.size(0), -1)
        x = self.conv4(x)
        x = self.prelu4(x)
        # detection
        label = nn.Sigmoid(self.conv5_1(x))
        offset = self.conv5_2(x)
        return label, offset


class Onet(nn.Module):
    def __init__(self):
        super(Onet, self).__init__()

        self.pre_layers = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size=3, stride=1),  # conv1
            nn.PReLU(),  # prelu1
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool1
            nn.Conv2d(32, 64, kernel_size=3, stride=1),  # conv2
            nn.PReLU(),  # prelu2
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool2
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # conv3
            nn.PReLU(),  # prelu3
            nn.MaxPool2d(kernel_size=2, stride=2),  # pool3
            nn.Conv2d(64, 128, kernel_size=2, stride=1),  # conv4
            nn.PReLU()  # prelu4

        )
        self.conv4 = nn.Linear(128 * 2 * 2, 256)  # conv4
        self.prelu4 = nn.PReLU()  # prelu4
        # detection
        self.conv5_1 = nn.Linear(256, 1)
        # bounding box regression
        self.conv5_2 = nn.Linear(256, 4)

    def forward(self, x):
        # backend
        x = self.pre_layer(x)
        x = x.view(x.size(0), -1)
        x = self.conv4(x)
        x = self.prelu4(x)
        # detection
        label = nn.Sigmoid(self.conv5_1(x))
        offset = self.conv5_2(x)
        return label, offset


if __name__ == '__main__':


    x = torch.Tensor(2, 3, 416, 416)
    # net = PNet()
    # cls, offset = net(x)
    # print(cls.shape)
    # print(offset.shape)
    # boxes=[]
    # with Image.open(img) as im:
    #     trans= transforms.Compose([transforms.ToTensor()])
    #     img = trans(im)
    #
    #     img.unsqueeze_(0)
    #     print(img.shape)
    #     net =PNet()
    #     cls,offset=net(img)
    #     idxs = torch.nonzero(torch.gt(cls, 0.6))
    #
    #
    #     print(idxs.shape)
    # print(offset.shape)
