import numpy as np
import utils
from PIL import Image
import os

LABEL_FILE_PATH = r"E:\cebela\Anno\list_bbox_celeba.txt"
IMAGE_PATH =""

def gen_simple(size):
    for i, line in enumerate(open(LABEL_FILE_PATH)):
        if i > 1:
            strs = list(filter(bool, line.split(" ")))
            filename = strs[0].strip()
            x1 = int(strs[1].strip())
            y1 = int(strs[2].strip())
            w = int(strs[3].strip())
            h = int(strs[4].strip())
            x2 = x1 + w
            y2 = y1 + h
            cx = x1 + w / 2
            cy = y1 + h / 2

            for _ in range(100):
                dx = np.random.uniform(-0.2, 0.2)
                dy = np.random.uniform(-0.2, 0.2)
                dw = np.random.uniform(-0.2, 0.2)
                dh = np.random.uniform(-0.2, 0.2)

                _cx = cx * (1 + dx)
                _cy = cy * (1 + dy)
                _w = w * (1 + dw)
                _h = h * (1 + dh)

                _x1 = _cx - _w / 2
                _y1 = _cy - _h / 2
                _x2 = _x1 + _w
                _y2 = _y1 + _h

                box = np.array([_x1, _y1, _x2, _y2, 0])
                boxes = np.array([[x1,y1,x2,y2,0]])

                im = Image.open(os.path.join(IMAGE_PATH,filename))
                
                _box =  utils.rect2squar(np.array([box]))[0]
                im = im.corp(_box[0:4])
                im.resize(size)

                iou = utils.iou(_box, boxes)

                if iou[0] > 0.65:#正样本
                    pass
                elif iou[0]>0.4: #部分样本
                    pass
                elif iou[0]<0.3:#负样本
                    pass



if __name__ == '__main__':
    gen_simple(12)
