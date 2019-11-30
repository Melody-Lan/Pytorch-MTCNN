import os

from torch.utils.data import DataLoader
import torch
from torch.autograd import Variable
from torch import nn
import torch.optim as optim

from simpling import FaceDataset


class Trainer():

    def __init__(self, net, save_path, File_path, iscuda=True):
        self.net = net
        self.save_path = save_path
        self.file_path = File_path
        self.iscuda = iscuda
        if iscuda:
            self.net.cuda()

        self.Conloss = nn.BCELoss()

        self.Offloss = nn.MSELoss()

        self.optimizer = optim.Adam(self.net.parameters())

        if os.path.exists(save_path):
            torch.load(save_path)

    def train(self):
        facedataset = FaceDataset(self.file_path)
        dataloader = DataLoader(facedataset, batch_size=100, shuffle=True, num_workers=4)
        while True:

            for i, (img_data_, category_, offset_) in enumerate(dataloader):
                if self.iscuda:
                    img_data_ = img_data_.cuda()
                    category_ = category_.cuda()
                    offset_ = offset_.cuda()
                _output_category, _output_offset = self.net(img_data_)

                output_category = _output_category.view(-1, 1)
                mask_category = torch.lt(category_, 2)
                output_category = torch.masked_select(output_category, mask_category)
                category = torch.masked_select(category_, mask_category)
                cls_loss = self.Conloss(output_category, category)

                mask_offset = torch.gt(category_, 0)

                offset = offset_[mask_offset]
                out_offset = _output_offset[mask_offset]
                offset_loss = self.Offloss(out_offset, offset)
                loss = cls_loss + offset_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                print(" loss:", loss.cpu().data.numpy(), " cls_loss:", cls_loss.cpu().data.numpy(), " offset_loss",
                      offset_loss.cpu().data.numpy())
            torch.save(self.net,self.save_path)
            print("save success")