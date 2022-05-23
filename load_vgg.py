import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision import models, transforms
from torchinfo import summary
from base_transform import BaseTransform
from ilsvrc_predictor import ILSVRCPredictor

print(torch.__version__)
print(torchvision.__version__)


net = models.vgg16(pretrained=True)
net.eval()  # 推論モード

# print(net)
# print(summary(net, (100, 3, 244, 244)))

image_file_path = 'data/goldenretriever-3724972_640.jpg'
img = Image.open(image_file_path)

resize = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
transform = BaseTransform(resize, mean, std)
img_transformed = transform(img)

predictor = ILSVRCPredictor()

inputs = img_transformed

out = net(inputs)
result = predictor.predict_max(out)

print(result)





