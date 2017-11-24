import sys, os
import numpy as np

sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show() 

(x_train, t_train), (x_test, t_test) = \
        load_mnist(flatten=True, normalize=False)

for i in range(5):
    img = x_train[i]
    label = t_train[i]
    print(label) 

    print(img.shape)
    img = img.reshape(28, 28)
    print(img.shape)
    img_show(img)
