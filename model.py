
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
from einops import rearrange

from module import temporal_block, transformer, SeparableConv2d, Conv2dWithConstraint

class HCANN(nn.Module):
    def __init__(self, fs, channels, dims, depth, heads,  num_classes, dim_initial, dropout=0.2, ):
        super(HCANN, self).__init__()
        self.dim_initial = dim_initial
        self.conv_T_1_1x16 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(1, 16), stride=1, bias=False),
            nn.BatchNorm2d(8, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU()
        )
        self.conv_T_2_1x16 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(1, 16), stride=1, bias=False),
            nn.BatchNorm2d(8, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU()
        )
        self.conv_T_3_1x8 = nn.Sequential(
            nn.ZeroPad2d((4, 3, 0, 0)),
            SeparableConv2d(in_channels=8, out_channels=16, kernel_size=(1, 8), stride=1, bias=False),
            nn.BatchNorm2d(16, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU()
        )
        self.conv_T_4_1x8 = nn.Sequential(
            nn.ZeroPad2d((4, 3, 0, 0)),
            SeparableConv2d(in_channels=16, out_channels=16, kernel_size=(1, 8), stride=1, bias=False),
            nn.BatchNorm2d(16, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU()
        )

        self.conv_S = nn.Sequential(
            Conv2dWithConstraint(in_channels=16, out_channels=16, kernel_size=(64, 1), bias=False, stride=(64, 1)),
            nn.BatchNorm2d(16, momentum=0.01, affine=True, eps=1e-3),
            nn.Hardswish()
        )

        self.AvgPooling1 = nn.AdaptiveAvgPool2d((64, self.dim_initial//4))
        self.AvgPooling2 = nn.AdaptiveAvgPool2d((64, self.dim_initial//32))
        self.dropout = nn.Dropout(dropout)

        self.transformer = nn.ModuleList([])
        self.transformer.append(transformer(depth=depth[0], dim=dims[0], heads=heads[0], hidden_feed=2, channel=8, dropout=dropout))
        self.transformer.append(transformer(depth=depth[1], dim=dims[1], heads=heads[1], hidden_feed=4, channel=16, dropout=dropout))


        self.fc = nn.Linear(dim_initial*16//32, 8)

    def forward(self, x):
        out = x
        # out = self.conv_channel64(x)
        out = rearrange(out, 'b c t -> b 1 c t')
        out = self.conv_T_1_1x16(out)
        out = self.conv_T_2_1x16(out)
        out = self.AvgPooling1(out)
        out = self.dropout(out)

        out = self.transformer[0](out)

        out = self.conv_T_3_1x8(out)
        out = self.conv_T_4_1x8(out)


        out = self.AvgPooling2(out)
        out = self.dropout(out)
        out = self.transformer[1](out)

        out = self.conv_S(out)


        out = out.flatten(start_dim=1)
        # out = self.fc2(self.fc(out))
        out = self.fc(out)
        return out

def get_variable(x):
    x = Variable(x)
    return x.cuda() if torch.cuda.is_available() else x
from Regularization import Regularization
def model_run(train_loader, test_loader, model, epoch_num, name, loss_func, optimizer, scheduler, num, scheduler_if=True):

    if torch.cuda.is_available():
        model = model.cuda()
    reg_loss = Regularization(model, weight_decay=0.1, p=2).cuda()

    all_train_loss, all_train_acc = [], []
    all_test_acc, all_test_loss = [], []
    all_predicted_train = []
    all_predicted_test = []
    for epoch in range(epoch_num):

        model.train()

        correct_train,total_train = 0,0
        running_loss = 0.0

        for i, (signals, labels) in enumerate(train_loader):
            signals = get_variable(signals)
            labels = get_variable(labels)

            outputs = model(signals)
            loss = loss_func(outputs, labels.long())+ reg_loss(model)

            # loss = loss_func[0](outputs, labels.long()) + reg_loss(model)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels.long().data).sum()
            all_predicted_train.extend(predicted)

            running_loss += loss.data.item()
        print('training loss of [%d/%d]: %.8f' % (epoch+1, epoch_num, running_loss))
        train_acc = 100 * correct_train / total_train
        print('train  Acc: %d %%' % (train_acc))
        if scheduler_if:
            scheduler.step()

        all_train_loss.append(running_loss/total_train)
        all_train_acc.append(train_acc.data.cpu().numpy())

        model.eval()
        correct, total = 0, 0
        running_test_loss = 0.0
        for signals, labels in test_loader:
            signals = get_variable(signals)
            labels = get_variable(labels)

            outputs = model(signals)
            loss = loss_func(outputs, labels.long())+ reg_loss(model)
            # loss = loss_func[0](outputs, labels.long())+ reg_loss(model)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.long().data).sum()

            running_test_loss += loss.data.item()
            all_predicted_test.extend(predicted)

        test_acc = 100 * correct / total
        print('test  Acc: %d %%' % (test_acc))
        all_test_acc.append(test_acc.data.cpu().numpy())
        all_test_loss.append(running_test_loss / total)

        x1 = range(0, epoch + 1)
        x2 = range(0, epoch + 1)
        y1 = all_train_acc
        y2 = all_train_loss
        y3 = all_test_acc
        y4 = all_test_loss

        fig = plt.subplot(2, 1, 1)
        plt.plot(x1, y1, c='red', label='train_acc')
        plt.plot(x1, y3, c='blue', label='test_acc')
        plt.ylabel('accuracy')
        plt.subplot(2, 1, 2)
        plt.plot(x2, y2, c='red', label='train_loss')
        plt.plot(x2, y4, c='blue', label='test_loss')
        plt.ylabel('loss')
        plt.savefig('result_path' + name + "_acc_loss" + str(num) + ".jpg")
        plt.close()


    return all_test_acc
