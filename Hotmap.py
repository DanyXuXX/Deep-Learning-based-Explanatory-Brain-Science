# encoding:utf-8
import torch as t
import torch.nn as nn
from torchvision import models
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import os.path

savedir = r'../hotmap/sgd_4_0.001_test/'
log = r'../hotmap/sgd_4_0.001.txt'


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.feature = t.nn.Sequential(
            t.nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0),
            t.nn.ReLU(),
            t.nn.MaxPool2d(kernel_size=3, stride=2),  # output_size = 27*27*96
            t.nn.Conv2d(96, 256, 5, 1, 2),
            t.nn.ReLU(),
            t.nn.MaxPool2d(3, 2),  # output_size = 13*13*256
            t.nn.Conv2d(256, 384, 3, 1, 1),
            t.nn.ReLU(),  # output_size = 13*13*384
            t.nn.Conv2d(384, 256, 3, 1, 1),
            t.nn.ReLU(),
            t.nn.MaxPool2d(3, 2)  # output_size = 6*6*256
        )

        # feed forward
        # RuntimeError: size mismatch, m1: [1000 x 6400], m2: [9216 x 4096]
        # All you have to care is b = c and you are done: m1: [a x b], m2: [c x d]
        # m1 is [a x b] which is [batch size x in features] in features.
        # It is not the input image size. When the input image is 96*96, it is 256. When the input image is 227*227, it is 9216.
        # m2 is [c x d] which is [ in features x out features]
        self.dense = t.nn.Sequential(
            t.nn.Linear(6400, 4096),
            t.nn.ReLU(),
            t.nn.Dropout(0.5),
            t.nn.Linear(4096, 4096),
            t.nn.ReLU(),
            t.nn.Dropout(0.5),
            t.nn.Linear(4096, 50)
        )

    def forward(self, x):
        feature_out = self.feature(x)
        res = feature_out.view(feature_out.size(0), -1)
        print("res" + str(res.shape))
        out = self.dense(res)
        return out


class FeatureExtractor(nn.Module):
    """
    1. Extract target layer features
    2. register target layer gradient
    """

    def __init__(self, model, target_layers):
        # def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        self.model = model
        for p in self.model.parameters():
            p.requires_grad = False
        # Define which layers you are going to extract
        # self.model_features = nn.Sequential(*list(self.model.children())[:4])
        # self.model_features = nn.Sequential(self.model.conv1, self.model.conv2, self.model.conv3, self.model.conv4,
        #                                     self.model.conv5)
        self.model_features = self.model.feature
        self.target_layers = target_layers
        self.gradients = list()

    def forward(self, x):
        return self.model_features(x)

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def get_gradients(self):
        return self.gradients

    def __call__(self, x):
        target_activations = list()
        self.gradients = list()
        for name, module in self.model_features._modules.items():  # Traverse each layer of the network
            x = module(x)  # input will pass through each layer traversed
            if name in self.target_layers:  # Set a condition, if it reaches the layer you specified, continue
                x.register_hook(self.save_gradient)  # Use hooks to record the gradient of the target layer
                target_activations += [x]  # Here only the features of the target layer are obtained
        x = x.view(x.size(0), -1)  # reshape into fully connected entry classifier
        x = self.model.dense(x)  # Enter classifier
        return target_activations, x,


def preprocess_image(img):
    """
    preprocessing layer
    standardize images
    """
    mean = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    preprocessed_img = img.copy()[:, :, ::-1]  # BGR > RGB

    # Standardized processing, processing all three layers of bgr
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - mean[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]

    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))  # transpose HWC > CHW
    preprocessed_img = t.from_numpy(preprocessed_img)  # totensor
    preprocessed_img.unsqueeze_(0)
    input = t.tensor(preprocessed_img, requires_grad=True)

    return input


def show_cam_on_image(img, mask, name):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)  # Use color space conversion to highlight the heatmap
    heatmap = np.float32(heatmap) / 255  # normalize
    cam = heatmap + np.float32(img)  # Overlay the heatmap onto the original image
    cam = cam / np.max(cam)
    cv2.imwrite(savedir + name, np.uint8(255 * cam))  # generate img

    # cam = cam[:, :, ::-1]  # BGR > RGB
    # plt.figure(figsize=(10, 10))
    # plt.imshow(np.uint8(255 * cam))


net = t.load('./srtp/2.0/models/sgd_4_0.001.pkl')


class GradCam():
    """
    GradCam mainly execution
    1. Extract features (call FeatureExtractor)
    2. Backpropagation to find the gradient of the target layer
    3. Realize the CAM diagram of the target layer
    """

    # def __init__(self, model, target_layer_names):
    def __init__(self, target_layer_names):
        self.model = net
        self.extractor = FeatureExtractor(self.model, target_layer_names)
        # self.extractor = FeatureExtractor(self.model, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input):
        features, output = self.extractor(input)  # The feature here corresponds to the output of the target layer, and output is the output of the image through the classification network.
        output.data
        one_hot = output.max()  # Take the largest value among 1000 classes
        print('one_hot',one_hot)
        # nn.Sequential(self.model.conv1, self.model.conv2, self.model.conv3, self.model.conv4,
        #               self.model.conv5)
        # self.model.features.zero_grad()  # Gradient clear
        # self.model.classifier.zero_grad()  # Gradient clear

        self.model.feature.zero_grad()
        self.model.dense.zero_grad()
        one_hot.backward(retain_graph=True)  # After backpropagation, in order to obtain the gradient of the target layer

        grad_val = self.extractor.get_gradients()[-1].data.numpy()
        # Call the function get_gradients() to get the gradient obtained by the target layer.

        target = features[-1]
        # Features is currently a list. We need to take out the output of the relu layer inside, which is the target layer shape we want (1, 512, 14, 14)
        target = target.data.numpy()[0, :]  # (1, 512, 14, 14) > (512, 14, 14)

        # weights = np.mean(grad_val, axis=(2, 3))[0, :]  # array shape (512, ) Find the weight of each layer of 512 layers of relu gradient
        weights = np.mean(grad_val, axis=(0, 1))
        cam = np.zeros(target.shape[1:])  # Make a blank map and fill in the values later
        # (14, 14)  shape(512, 14, 14)tuple  Index[1:] means starting from 14

        # The for loop method multiplies the average weight by each feature map of the target layer, and adds it to the blank map just generated.
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
            # w * target[i, :, :]
            # target[i, :, :] = array:shape(14, 14)
            # w = weighted average of 512, shape(512, )
            # Each mean value is multiplied by the feature map of the target.
            # Place it on the blank 14*14 (cam)
            # Eventually the 14*14 empty map will be filled

        cam = cv2.resize(cam, (224, 224))  # Enlarge the 14*14 featuremap back to 224*224
        # print(type(cam))
        # print(cam.shape)
        print(cam)
        # cam = cam - np.min(cam)
        # cam = cam / np.max(cam)
        # for i, row in enumerate(cam):
        #     for j, val in enumerate(row):
        #         if val > 0.005:
        #             cam[i][j] = 0.005
        print(str(np.max(cam)), "      ", str(np.min(cam)))
        f1 = open(log, 'r+')
        f1.read()
        f1.write(str(cam) + "\n")
        f1.write(str(np.max(cam)) + "      " + str(np.min(cam)) + "\n")
        # cam = cam - np.min(cam)
        # cam = cam / np.max(cam)
        return cam


# grad_cam = GradCam(model=net, target_layer_names=["7"])
# for i in range(13):
#     grad_cam = GradCam(target_layer_names=["%d" % i])
#     print(i)
#
#     img = cv2.imread('./dataset/test/up/Omega-02-Jun-2019-2-53176.jpg')
#     img = np.float32(cv2.resize(img, (227, 227))) / 255  # In order to put it into the 224*224 required by vgg16, first scale and normalize it.
#     input = preprocess_image(img)
#     mask = grad_cam(input)
#     show_cam_on_image(img, mask, i, "up")
#
#     img = cv2.imread('./dataset/test/stay/Patamon-10-Jun-2019-1-123255.jpg')
#     img = np.float32(cv2.resize(img, (227, 227))) / 255
#     input = preprocess_image(img)
#     mask = grad_cam(input)
#     show_cam_on_image(img, mask, i, "stay")
#
#     img = cv2.imread('./dataset/test/right/Patamon-10-Jun-2019-1-43386.jpg')
#     img = np.float32(cv2.resize(img, (227, 227))) / 255
#     input = preprocess_image(img)
#     mask = grad_cam(input)
#     show_cam_on_image(img, mask, i, "right")
#
#     img = cv2.imread('./dataset/test/down/Patamon-10-Jun-2019-1-140615.jpg')
#     img = np.float32(cv2.resize(img, (227, 227))) / 255
#     input = preprocess_image(img)
#     mask = grad_cam(input)
#     show_cam_on_image(img, mask, i, "down")


rootdir = r'../feature_eyes/'
hp_pic_list = os.listdir(rootdir)
grad_cam = GradCam(target_layer_names=["10"])
for htmp in hp_pic_list:
    # print(htmp[0:23])
    if htmp[0:23] == '12-1-Omega-25-Jun-2019_':
        currentPath = os.path.join(rootdir, htmp)
        # currentPath = r'./1-1-Omega-01-Jun-2019.mp4_1.jpg'
        print('the fulll name of the file is :' + currentPath)
        img = cv2.imread(currentPath)
        img = np.float32(cv2.resize(img, (224, 224))) / 255
        input = preprocess_image(img)
        mask = grad_cam(input)
        # show_cam_on_image(img, mask, htmp)
