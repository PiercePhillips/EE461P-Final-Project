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

alphabet_vgg = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','del','nothing','space']

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
                vgg_frame, tensor = preprocess(frame)
                preds = predict(tensor)
                print(f'{alphabet_vgg[torch.argmax(preds)]}')

                # display frame
                cv2.imshow('output', frame)
                cv2.imshow('resized_output', cv2.resize(vgg_frame, (224, 224)))

                cv2.waitKey(1)

        except KeyboardInterrupt:
            # clean up
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)
                

def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[0], img.shape[1]
    center = (img.shape[0] / 2, img.shape[1] / 2)
    x = center[1] - w/2
    y = center[0] - h/2
    crop_img = img[int(y):int(y+h), int(x):int(x+w)]

    resized_img = cv2.resize(crop_img, (224, 224))

    tensor = resized_img / 255.0
    tensor = np.moveaxis(tensor, -1, 0)
    tensor = np.expand_dims(tensor, axis=0)

    return resized_img, tensor


def predict(img):
    # center image
    tensor = torch.from_numpy(img).float()
    preds = model_vgg(tensor)

    return preds


if __name__ == '__main__':
    # model init
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #VGG-16 model
    model_vgg = models.vgg16(pretrained=False)
    number_features = model_vgg.classifier[6].in_features
    model_vgg.classifier[6] = nn.Linear(number_features, 29)

    model_vgg.load_state_dict(torch.load('model_vgg.pth'))
    model_vgg.eval()

    camera() # main camera loop