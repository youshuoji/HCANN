import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as Data
from einops import rearrange


from model import HCANN, model_run
from utils import make_dataset

# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

NUM_CLASS = 8

batch_size = 16
lr = 0.001
dropout = 0.5
# weight_decay = 1
epoch_num = 10000
fs = 1024
channels = [8, 16, 16, 8, 8, 16]
dim_initial = fs
depth = [6, 4]
heads = [4, 2]
dims = [dim_initial//4, dim_initial//32]
milestones = [4000, 8000]
gamma = 2.5
name = 'name'
all_acc = []
path = 'path_to_data'


data_train, label_train, data_test, label_test = make_dataset(load_path=path + name + '.npz', standardization=False,
                                                                t_ratio=0.7)
data_train, data_test, label_train, label_test = torch.Tensor(np.asarray(data_train)), torch.Tensor(
    np.asarray(data_test)), torch.Tensor(np.asarray(label_train)), torch.Tensor(np.asarray(label_test))

train_dataset = Data.TensorDataset(data_train, label_train)
test_dataset = Data.TensorDataset(data_test, label_test)

train_loader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = Data.DataLoader(dataset=test_dataset, batch_size=batch_size * 2, shuffle=True)
model = HCANN(fs=fs, channels=channels, dims=dims, dim_initial=dim_initial, depth=depth, heads=heads, num_classes=NUM_CLASS, dropout=dropout)

loss_func = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.001)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
test_acc = model_run(train_loader, test_loader, model, epoch_num, name, loss_func, optimizer, scheduler, scheduler_if=True, num=1)
print(test_acc)


