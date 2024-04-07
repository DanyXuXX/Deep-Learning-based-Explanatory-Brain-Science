import cv2
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from PIL import Image
from torch import nn
import torch as t
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils,models
import torch.optim as optim
from torch.autograd import Variable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
heatmap_list=[]

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
        # m1 is [a x b] which is [batch size x in features] in features. It is not the input image size. When the input image is 96*96, it is 256. When the input image is 227*227, it is 9216.
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
        # print("res" + str(res.shape))
        out = self.dense(res)
        return out

# model=CNN()

model = t.load('sgd_4_0.001.pkl')
# print(model)
# print(model)
# print(model.features)
# model.load_state_dict(torch.load('../models/sgd_4_0.001.pkl'))
# print(model.feature)

def draw_CAM(model, img_path, save_path, transform=None, visual_heatmap=False):

    # Image loading & preprocessing
    img = Image.open(img_path).convert('RGB')
    img = transform(img)
    img = img.unsqueeze(0)

    # img = img.unsqueeze(0)
    # print(transform)
    # Get the feature/score of the model output
    model.eval()
    # feature_2=model.classifier(img)
    # print(img.shape) #
    feature_1=model.feature(img)
    # print('feature1',feature_1[0].shape)
    res = feature_1.view(feature_1.size(0), -1)
    output = model.dense(res)
    # print('out',output)

    # In order to be able to read the auxiliary function defined by the intermediate gradient
    def extract(g):
        global features_grad
        features_grad = g

    # The output score corresponding to the category with the highest prediction score
    pred = torch.argmax(output).item()
    # print(pred)
    pred_class = output[0][pred]
    # pred_class = output[:, pred]
    # print(pred_class)

    feature_1.register_hook(extract)
    # pred_class.backward()  # calculate grad
    pred_class.backward()
    grads = features_grad  # get grad
    # print('grads',grads)
    pooled_grads = torch.nn.functional.adaptive_avg_pool2d(grads, (1, 1))
    # print('pooled',pooled_grads.shape)
    # The batch size here defaults to 1, so the 0th dimension (batch size dimension) is removed.
    pooled_grads = pooled_grads[0]
    feature_1 = feature_1[0]
    feature_1 = feature_1.permute(1, 2, 0) # change into（6，6，256）

    # print('pooled[1]',pooled_grads[1],pooled_grads[2])
    for i in range(256):
        # features[i, ...] *= pooled_grads[i, ...]
        feature_1[:,:,i] *= pooled_grads[i]

    heatmap = feature_1.detach().numpy()
    # print('0',heatmap.shape)  #（6，6，256）


    heatmap = np.mean(heatmap, axis=-1)
    # print('1,',heatmap.shape)  # （6，6）
    heatmap = np.maximum(heatmap, 0)
    # print('2',heatmap.shape)
    heatmap /= np.max(heatmap)
    # print('3',heatmap) # （6，6）

    # print('heatmap',heatmap.shape)

    heatmap_list.append(heatmap)
    # Visualizing raw heatmaps
    if visual_heatmap:
        plt.matshow(heatmap)
        plt.show()

    img = cv2.imread(img_path)  # Load original image with cv2
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # Resize the heatmap to the same size as the original image
    heatmap = np.uint8(255 * heatmap)  # Convert heatmap to RGB format
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # Apply heatmap to original image
    superimposed_img = heatmap * 0.4 + img  # 0.4 here is the heat map intensity factor
    # cv2.imwrite(save_path, superimposed_img)  # Save image to hard drive
    # cv2.imwrite(save_path, superimposed_img)
Mytransform = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224),
                                                transforms.ToTensor()])

df=pd.read_csv('sequence.txt')
for i in df['path']:
    # print(i[-9::])
    #A print(i)
    save_path='./save_sequence_CNN_4_0.001/'+i[-9::]
    # print(save_path)
    draw_CAM(model,i,save_path,Mytransform)
    # draw_CAM(model,'Omega-02-Jul-2019-1-10014.jpg','./',Mytransform)
# draw_CAM(model,'../../data/dataset/test/left/Omega-02-Jul-2019-1-10066.jpg','./',Mytransform)
# draw_CAM(model,'Omega-02-Jul-2019-1-10014.jpg','./',Mytransform)

delta=(len(heatmap_list)-1)*['']

for i in range(len(heatmap_list)-1):
    delta[i] = heatmap_list[i+1] - heatmap_list[i]

# print(delta)
mean_list=[]
for i in range(len(delta)):
    mean=np.mean(delta[i])
    mean_list.append(mean)

file = open('delta-mean-cnn.txt', 'w')
file.write(str(mean_list))
file.close()

'''
heatmap_list[6].flatten()
plt.hist(heatmap_list[6],bins=10)
plt.title('hist_cnn_6')
plt.savefig("hist_cnn_6.jpg")
plt.show()
'''
'''
mean_list=[]
for i in range(len(heatmap_list)):
    var=np.mean(heatmap_list[i])
    mean_list.append(var)
file = open('mean_cnn.txt', 'w')
file.write(str(mean_list))
file.close()
la = range(len(mean_list))
plt.plot(la, mean_list, 'r', label='mean-cnn')
'''


