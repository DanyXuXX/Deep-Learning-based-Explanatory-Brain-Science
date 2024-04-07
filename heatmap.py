import cv2
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils,models
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
heatmap_list=[]
class ConvLSTM(nn.Module):
    def __init__(self):
        super(ConvLSTM, self).__init__()
        net = models.alexnet(pretrained=True)
        net.classifier = nn.Sequential()
        self.features = net
        self.rnn = nn.LSTM(input_size=256 * 6 * 6, hidden_size=64, num_layers=2, batch_first=True)
        self.out = nn.Linear(64, 5)


    def forward(self, x):

        B=x.size(0)
        x=x.view(B,3,224,224)

        output=self.features(x)
        output=output.view(B,-1).transpose(0,1).contiguous().view(256*6*6,B,1)
        output=output.permute(1,2,0)


        h0 = torch.zeros(2, x.size(0), 64).to(device)
        c0 = torch.zeros(2, x.size(0), 64).to(device)

        # 前向传播LSTM$
        out, _ = self.rnn(output, (h0, c0))  # 输出大小 (batch_size, seq_length, hidden_size)
        # print(out.shape)
        # print('1:',out.size())
        # 解码最后一个时刻的隐状态
        out = self.out(out[:, -1, :])
        # print('2:',out.size())
        # print(out.size())
        return out

model=ConvLSTM()
# print(model)
# print(model.features)
model.load_state_dict(torch.load('convRNN.pth'))
# print(model.features.features)

def draw_CAM(model, img_path, save_path, transform=None, visual_heatmap=False):

    # 图像加载&预处理
    img = Image.open(img_path).convert('RGB')
    img = transform(img)
    img = img.unsqueeze(0)

    # img = img.unsqueeze(0)
    # print(transform)
    # 获取模型输出的feature/score
    model.eval()
    # feature_2=model.classifier(img)
    # print(img.shape) #
    feature_1=model.features.features(img)
    # print('feature1',feature_1[0].shape) #
    feature_2 = feature_1.view(1, -1).transpose(0, 1).contiguous().view(256 * 6 * 6, 1, 1)
    feature_2 = feature_2.permute(1, 2, 0)
    # print(feature_1.shape) #

    h0 = torch.zeros(2, img.size(0), 64).to(device)
    c0 = torch.zeros(2, img.size(0), 64).to(device)

    # 前向传播LSTM$
    features, _ = model.rnn(feature_2, (h0, c0))
    # features = model.rnn(feature_1)
    # print(type(features[1]),type(features))
    # print(features.shape)
    output = model.out(features)
    # print('out',output)

    # 为了能读取到中间梯度定义的辅助函数
    def extract(g):
        global features_grad
        features_grad = g

    # 预测得分最高的那一类对应的输出score
    pred = torch.argmax(output).item()
    # print(pred)
    pred_class = output[0][0][pred]
    # pred_class = output[:, pred]
    # print(pred_class)

    feature_1.register_hook(extract)
    # pred_class.backward()  # 计算梯度
    pred_class.backward()
    grads = features_grad  # 获取梯度
    # print('grads',grads.shape)
    pooled_grads = torch.nn.functional.adaptive_avg_pool2d(grads, (1, 1))
    # print('pooled',pooled_grads.shape)
    # 此处batch size默认为1，所以去掉了第0维（batch size维）
    pooled_grads = pooled_grads[0]
    feature_1 = feature_1[0]
    feature_1 = feature_1.permute(1, 2, 0) # 变成（6，6，256）

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
    heatmap_list.append(heatmap)
    print('heatmap',heatmap.shape)


    # 可视化原始热力图
    if visual_heatmap:
        plt.matshow(heatmap)
        plt.show()

    img = cv2.imread(img_path)  # 用cv2加载原始图像
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
    heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
    superimposed_img = heatmap * 0.4 + img  # 这里的0.4是热力图强度因子
    # cv2.imwrite(save_path, superimposed_img)  # 将图像保存到硬盘
    # cv2.imwrite(save_path, superimposed_img)

    os.makedirs(save_path, exist_ok=True)  # Ensure that the directory exists
    cv2.imwrite(os.path.join(save_path, os.path.basename(img_path)), superimposed_img)

Mytransform = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224),
                                      transforms.ToTensor()])






df=pd.read_csv('sequence.txt')
for i in df['path']:
    # print(i[-9::])
    # print(i)
    save_path='./save_sequence/'+i[-9::]
    print("save_path:", os.path.abspath('save_sequence'))
    # print(save_path)
    draw_CAM(model,i,save_path,Mytransform)
    # draw_CAM(model,'Omega-02-Jul-2019-1-10014.jpg','./',Mytransform)

delta=(len(heatmap_list)-1)*['']

for i in range(len(heatmap_list)-1):
    delta[i] = heatmap_list[i+1] - heatmap_list[i]

print(delta)
mean_list=[]
for i in range(len(delta)):
    mean=np.var(delta[i])
    mean_list.append(mean)

file = open('delta-var.txt', 'w')
file.write(str(mean_list))
file.close()

'''
heatmap_list[6].flatten()
plt.hist(heatmap_list[6],bins=10)
plt.title('hist_cnn_rnn_6')
plt.savefig("hist_cnn_rnn_6.jpg")
plt.show()
'''
'''
file=open('data.txt','w')
file.write(str(heatmap_list))
file.close()
'''
'''
mean_list=[]
for i in range(len(heatmap_list)):
    var=np.var(heatmap_list[i])
    mean_list.append(var)
file = open('var.txt', 'w')
file.write(str(mean_list))
file.close()'''