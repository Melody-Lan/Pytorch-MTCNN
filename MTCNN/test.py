import os
from PIL import Image
import utils
import numpy as np
import traceback

anno_src = r"D:\test\MTCNN\data\Anno\list_bbox_celeba.txt"
img_dir = r"D:\test\MTCNN\data\img"

image_file = os.path.join(img_dir, "000001.jpg")
with Image.open(image_file) as img:
    print(img.size)