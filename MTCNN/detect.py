import torch
from PIL import Image
from PIL import ImageDraw
import numpy as np
import utils

import nets

from torchvision import transforms
import time


class Detector:

    def __init__(self, pnet_param="./param/pnet.pt", rnet_param="./param/rnet.pt", onet_param="./param/onet.pt",
                 isCuda=True):
        self.isCuda = isCuda

        self.pnet = nets.PNet()
        self.rnet = nets.RNet()
        self.onet = nets.ONet()

        if self.isCuda:
            self.pnet.cuda()
            self.rnet.cuda()
            self.onet.cuda()
        torch.load(pnet_param)
        torch.load(rnet_param)
        torch.load(onet_param)
        self.pnet.eval()
        self.rnet.eval()
        self.onet.eval()
        self.__image_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def detect(self, image):
        start_time = time.time()

        boxre = self.Pnet_detect(image)

    def Pnet_detect(self, img):
        boxes = []

        w, h = img.size
        min_side_len = min(w, h)

        scale = 1

        while min_side_len > 12:
            img_data = self.__image_transform(img)
            if self.isCuda:
                img_data = img_data.cuda()
            img_data.unsqueeze_(0)

            _cls, _offest = self.pnet(img_data)

            cls, offest = _cls[0][0].cpu().data, _offest[0].cpu().data

            idxs = torch.nonzero(torch.gt(cls, 0.6))

            for idx in idxs:
                boxes.append(self.__box(idx, offest, cls[idx[0], idx[1]], scale))
            scale = 0.8
            _w = int(w * scale)
            _h = int(h * scale)
            img = img.resize((_w, _h))

            min_side_len = min(_w, _h)
        return utils.nms(np.array(boxes), 0.5)

    def __box(self, start_index, offset, cls, scale, stride=2, side_len=12):
        _x1 = (start_index[1] * stride) / scale
        _y1 = (start_index[0] * stride) / scale
        _x2 = (start_index[1] * stride + side_len) / scale
        _y2 = (start_index[0] * stride + side_len) / scale

        ow = _x2 - _x1
        oh = _y2 - _y1

        _offset = offset[:, start_index[0], start_index[1]]
        x1 = _x1 + ow * _offset[0]
        y1 = _y1 + oh * _offset[1]
        x2 = _x2 + ow * _offset[2]
        y2 = _y2 + oh * _offset[3]

        return [x1, y1, x2, y2, cls]
    def Rnet_detect(self,pboxes,image):

        img_data_boxes=[]

        pboxes = utils.convert_to_square(pboxes)

        for _box in pboxes:
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            img = image.crop((_x1, _y1, _x2, _y2))
            img = img.resize((24, 24))
            img_data = self.__image_transform(img)
            img_data_boxes.append(img_data)

        img_dataset = torch.stack(img_data_boxes)
        if self.isCuda:
            img_dataset=img_dataset.cuda()
        img_dataset = self.__image_transform(img_dataset)

        cls,offset= self.rnet(img_dataset)

        cls = cls.cpu().data.numpy()
        offset = offset.cpu().data.numpy()
        boxes = []
        idxs, _ = np.where(cls > 0.6)
        for idx in idxs:
            _box = pboxes[idx]
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            ow = _x2 - _x1
            oh = _y2 - _y1

            x1 = _x1 + ow * offset[idx][0]
            y1 = _y1 + oh * offset[idx][1]
            x2 = _x2 + ow * offset[idx][2]
            y2 = _y2 + oh * offset[idx][3]

            boxes.append([x1, y1, x2, y2, cls[idx][0]])

        return utils.nms(np.array(boxes), 0.5)


    def __onet_detect(self, image, rnet_boxes):

        _img_dataset = []
        _rnet_boxes = utils.convert_to_square(rnet_boxes)
        for _box in _rnet_boxes:
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            img = image.crop((_x1, _y1, _x2, _y2))
            img = img.resize((48, 48))
            img_data = self.__image_transform(img)
            _img_dataset.append(img_data)

        img_dataset = torch.stack(_img_dataset)
        if self.isCuda:
            img_dataset = img_dataset.cuda()

        _cls, _offset = self.onet(img_dataset)

        cls = _cls.cpu().data.numpy()
        offset = _offset.cpu().data.numpy()

        boxes = []
        idxs, _ = np.where(cls > 0.97)
        for idx in idxs:
            _box = _rnet_boxes[idx]
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            ow = _x2 - _x1
            oh = _y2 - _y1

            x1 = _x1 + ow * offset[idx][0]
            y1 = _y1 + oh * offset[idx][1]
            x2 = _x2 + ow * offset[idx][2]
            y2 = _y2 + oh * offset[idx][3]


            boxes.append([x1, y1, x2, y2, cls[idx][0]])

        return utils.nms(np.array(boxes), 0.7, isMin=True)