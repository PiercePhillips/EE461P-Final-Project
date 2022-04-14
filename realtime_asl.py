import sys
import argparse
import time
import cv2
import numpy as np
import pandas as pd

import torch
from torchvision.models import detection
import torch.nn as nn
import torch.nn.functional as F
import PIL.Image as Image
from torchvision.transforms import ToTensor, ToPILImage

alphabet = ['a','b','c','d','e','f','g','h','i','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y']

# define our model
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


def camera():
    cap = cv2.VideoCapture(0)

    # set image resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    fps = 10
    prev_t = 0
    # capture video frames
    while True:
        try:
            delta_t = time.time() - prev_t
            res, frame = cap.read()

            if delta_t > 1.0/fps:
                prev_t = time.time()

                # generate prediction
                mnist_frame = preprocess(frame, (784))
                preds = predict(mnist_frame)
                print(f'{alphabet[torch.argmax(preds)]}')

                # display frame
                cv2.imshow('output', frame)
                cv2.imshow('resized_output', cv2.resize(mnist_frame, (300, 300)))

                cv2.waitKey(1)

        except KeyboardInterrupt:
            # clean up
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)
                

def preprocess(img, dims):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[0], img.shape[1]
    center = (img.shape[0] / 2, img.shape[1] / 2)
    x = center[1] - w/2
    y = center[0] - h/2
    crop_img = img[int(y):int(y+h), int(x):int(x+w)]

    resized_img = cv2.resize(crop_img, (28, 28))
    img = img.reshape((784,)) / 255.0

    return resized_img


def predict(img):
    # center image
    tensor = torch.from_numpy(img).float()
    preds = model(tensor)

    return preds


if __name__ == '__main__':
    # model init
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MNist_Net()
    model.load_state_dict(torch.load('model_mnist.pth'))
    model.eval()

    camera() # main camera loop