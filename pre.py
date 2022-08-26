
from torchvision.models import resnet50
from requests import get
from torch import nn as nn
from torch import load
from os import environ
import numpy as np
from PIL import Image
from torchvision import transforms
from io import BytesIO
from sys import argv
environ['KMP_DUPLICATE_LIB_OK'] = 'True'
def softmax(x):
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x, 0)
    return softmax_x

with open('dir_label.txt', 'r', encoding='utf-8') as f:
    labels = f.readlines()
    labels = list(map(lambda x:x.strip().split('\t'), labels))
def image_get(path):
    try:
        image = Image.open(path)
    except:
        yzmdata = get(path)
        tempIm = BytesIO(yzmdata.content)
        image = Image.open(tempIm)
    return image
if __name__ == "__main__":

    model = resnet50(pretrained=False)
    fc_inputs = model.fc.in_features
    model.fc = nn.Linear(fc_inputs, 214)
        # model = model.cuda()
        # 加载训练好的模型
    checkpoint = load('model_best_checkpoint_resnet50.pth.tar',map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    path = argv[1]
    img = image_get(path)
    # img = Garbage_Loader.padding_black(img)
    transform = transforms.Compose([
    transforms.Resize(224),
    # 从图像中心裁切224x224大小的图片
    transforms.ToTensor(),
    ])
    img=transform(img).unsqueeze(0)
    pred = model(img)
    pred = pred.data.cpu().numpy()[0]
    score = softmax(pred)
    pred_id = np.argmax(score)
        # plt.imshow(src)
    print('预测结果：', labels[pred_id][0])
        # plt.show()
     

