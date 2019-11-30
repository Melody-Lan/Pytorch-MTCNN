import nets
import train
if __name__ == '__main__':
    net = nets.ONet()

    trainer = train.Trainer(net, './param/onet.pt', r"C:\celeba4\48")
    trainer.train()