from PIL import Image
from torchvision import transforms
import numpy as np

h = 376
w = 1241
th, tw = 352, 1216
i = h - th
j = int(round((w - tw) / 2.))
print(i, j, th, tw)

img = Image.open('std_result/0000002700.png')
# 300x300
img1 = np.array(img, dtype=int)
print(img1.shape)
print(img1)
transform_1 = transforms.BottomCrop((351, 1213))
img_1 = transform_1(img)
img1_1 = np.array(img_1, dtype=int)
print(img1_1.shape)
print(img1_1)
