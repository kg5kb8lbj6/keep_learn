import os
from pickletools import optimize
import torch
import math
import load_data as ld
from torch import nn
from cnn import Net
from torchvision import transforms
from torch.utils.data import DataLoader
from torchsummary import summary
from torch.autograd import Variable
import matplotlib.pyplot as plt


epoch_list = []
loss_list = []
acc_list = []

batch_size = 128
image = r'D:\learn\MNIST_Dataset\train_images'
def train():
    os.makedirs('./data', exist_ok = True)
    if True:
        ld.image_root(image, './data/total.txt')
        ld.shuffle_split('./data/total.txt', './data/trainFile.txt', './data/valFile.txt')
    
    train_data = ld.MyDataset('./data/trainFile.txt', transform = transforms.ToTensor())
    val_data = ld.MyDataset('./data/valFile.txt', transform = transforms.ToTensor())
    train_loader = DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(dataset = val_data, batch_size = batch_size)
    
    model = Net()
    #summary(model, (3, 28, 28))
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01, weight_decay = 1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 20], 0.1)
    loss_func = nn.CrossEntropyLoss()
    epochs = 30


    for epoch in range(epochs):
        epoch_list.append(epochs)
        # training-----------------------------
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        for batch, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = Variable(batch_x), Variable(batch_y)
            out = model(batch_x)
            loss = loss_func(out, batch_y)
            train_loss += loss
            pred = torch.max(out, 1)[1]
            train_correct = (pred == batch_y).sum()
            train_acc += train_correct.item()
            print('epoch: %2d/%d batch %3d/%d  Train Loss: %.3f, Acc: %.3f'
                % (epoch + 1, epochs, batch, math.ceil(len(train_data) / batch_size),
                    loss.item(), train_correct.item() / len(batch_x)))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        loss_list.append(train_loss / (math.ceil(len(train_data) / batch)))
        acc_list.append(train_acc / (len(train_data)))

        print('Train Loss:%.6f, Acc: %.3f'%(train_loss / (math.ceil(len(train_data) / batch)),
                train_acc / (len(train_data))))
        
        # eval------------------------------
        model.eval()
        eval_loss = 0.0
        eval_acc = 0.0
        for batch_x, batch_y in val_loader:
            batch_x , batch_y = Variable(batch_x), Variable(batch_y)
            out = model(batch_x)
            loss = loss_func(out, batch_y)
            eval_loss += loss.item()
            pred = torch.max(out, 1)[1]
            num_correct = (pred == batch_y).sum()
            eval_acc += num_correct.item()
        print('Val Loss: %.6f, Acc: %.3f' % (eval_loss / (math.ceil(len(val_data)/batch_size)),
                                            eval_acc / (len(val_data))))
        # save model --------------------------------
        if (epoch + 1) % 1 == 0:
            # torch.save(model, 'output/model_' + str(epoch+1) + '.pth')
            torch.save(model.state_dict(), r'D:\learn\mnist_model\params_' + str(epoch + 1) + '.pth')
            #to_onnx(model, 3, 28, 28, 'params.onnx')

            













if __name__ == '__main__':
    train()
    plt.plot(epoch_list, loss_list)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()


