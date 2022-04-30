import sys
import argparse
import time
import cv2
import numpy as np
import pandas as pd

import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import PIL.Image as Image
from torchvision.transforms import ToTensor, ToPILImage

alphabet_mnist = ['a','b','c','d','e','f','g','h','i','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y']

# define our models
class MNist_Net(nn.Module):
    def __init__(self):
        super(MNist_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 256, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(256, 24, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        #x = F.avg_pool2d(x, 4)
        return F.log_softmax(x)

class MNist_Net3(nn.Module):
    def __init__(self):
        super(MNist_Net3, self).__init__()
        self.cov = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(32,64, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(64),
        
            nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(128 ,128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(256,256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2304,1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512,24)
        )
    
    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.cov(x)
        x = self.linear(x)
        return F.log_softmax(x)


def camera():
    cap = cv2.VideoCapture(0)

    # set image resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    fps = 1
    prev_t = 0
    # capture video frames
    while True:
        try:
            delta_t = time.time() - prev_t
            res, frame = cap.read()

            if delta_t > 1.0/fps:
                prev_t = time.time()

                # generate prediction
                mnist_frame, tensor = preprocess_mnist(frame, (784,))
                preds = predict(tensor)
                print(f'{alphabet_mnist[torch.argmax(preds)]}')

                # display frame
                cv2.imshow('output', frame)
                cv2.imshow('resized_output', cv2.resize(mnist_frame, (300, 300)))

                cv2.waitKey(1)

        except KeyboardInterrupt:
            # clean up
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)
                

def preprocess_mnist(img, dims):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[0], img.shape[1]
    center = (img.shape[0] / 2, img.shape[1] / 2)
    x = center[1] - w/2
    y = center[0] - h/2
    crop_img = img[int(y):int(y+h), int(x):int(x+w)]

    resized_img = cv2.resize(crop_img, (28, 28))
    tensor = resized_img.reshape((784,)) / 255.0

    return resized_img, tensor


def predict(img):
    # center image
    tensor = torch.from_numpy(img).float()
    preds = model(tensor)

    return preds


if __name__ == '__main__':
    # model init
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # MNIST model with our own network
    model = MNist_Net3()
    model.load_state_dict(torch.load('model_weights/model4.pth'))
    model.eval()

    camera() # main camera loop