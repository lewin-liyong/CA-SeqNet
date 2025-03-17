import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import random
from data_preprocess import preProc
from core import VGGWithCBAM1D_channel
from torch.autograd import Variable
from tools import local_time
import time
import math
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['STSong']
plt.rcParams['axes.unicode_minus'] = False

sys.path.append('./data')
sys.path.append('core')
sys.path.append('..')


# Set random seeds to ensure the consistency of model initialization and small batch loading
def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

torch.cuda.set_device(2)
set_seed(42)  # Set random seed

# Dataset path
dataset_train_path = r'/RAID5/projects/ly/class/train'
dataset_test_path = r'/RAID5/projects/ly/class/test'

# Data preprocessing
model = 'SeqNet'
train_data = preProc(data_dir=dataset_train_path, model=model)
test_data = preProc(data_dir=dataset_test_path, model=model)

# Training parameters
MAX_EPOCH = 100
BATCH_SIZE = 16
LR = 0.001
log_interval = 4
val_interval = 1

# Data loader
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True,
                          worker_init_fn=lambda _: np.random.seed(42))
valid_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True,
                          worker_init_fn=lambda _: np.random.seed(42))

# Model and optimizer
net = VGGWithCBAM1D_channel()
if torch.cuda.is_available():
    net.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=LR, betas=(0.9, 0.99))
optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=1e-4)

# Learning rate decay, reduce the learning rate every 10 epochs, gamma determines the multiple of learning rate reduction
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

accur_temp = -1
order = str(time.time())
output_dir = os.getcwd() + '/outputs' + '/' + order
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
file = open(output_dir + '/' + 'train_log.txt', 'w')

# Record parameter log
log_1 = 'Parameters:img_size=' + str(train_data.img_size) + ', is_small_batch_test=' + str(train_data.small_batch_test) \
        + ', num_of_gpu=' + str(torch.cuda.device_count())
log_2 = ', MAX_EPOCH=' + str(MAX_EPOCH) + ', BATCH_SIZE=' + str(BATCH_SIZE) + ', LR=' + str(LR) + '-->' + model
file.write(local_time() + log_1 + log_2 + '\n')
file.flush()
print(local_time() + log_1 + log_2)

# Initialize the storage of various indicators
loss_train_list = []
loss_val_list = []
precision_list = [[] for i in range(4)]
recall_list = [[] for i in range(4)]
Macro_F1_list = []
accuracy_list = []
# Training process
for epoch in range(MAX_EPOCH):
    # Start recording time
    iter_time = [0, 0]
    loss_item = 0.0
    correct = 0.
    F1_score = 0.
    total = 0.
    predic_count = [0. for i in range(4)]
    sampl_count = [0. for i in range(4)]
    correct_count = [0. for i in range(4)]

    net.train()
    for i, data in enumerate(train_loader):
        if not iter_time[1]:
            iter_time[0] = time.time()

        input, label = data
        input = Variable(input[:, None, :]).float()
        label = Variable(label).long()

        if torch.cuda.is_available():
            input = input.cuda()
            label = label.cuda()

        optimizer.zero_grad()
        out = net(input)
        loss_train = criterion(out, label)
        loss_train.backward()
        optimizer.step()

        loss_item += loss_train.data.item()
        if (i + 1) % log_interval == 0:
            iter_time[1] = time.time()
            log_time = ', iter_time: ' + str(round(iter_time[1] - iter_time[0], 4))
            iter_time[0] = iter_time[1]
            log_1 = '[epoch: ' + str(epoch + 1) + '/' + str(MAX_EPOCH) + ']. iter: ' + str(i + 1) + '/' + str(
                math.ceil(len(train_data.npy_list) / BATCH_SIZE))
            log_2 = 's, loss: ' + str(round(loss_item / (i + 1), 7))
            print(local_time() + log_1 + log_time + log_2)
            file.write(local_time() + log_1 + log_time + log_2 + '\n')
            file.flush()

    loss_train_list.append(loss_item / (i + 1))

    # Learning rate update
    scheduler.step()

    net.eval()
    loss_item2 = 0.
    correct = 0.
    total = 0.
    with torch.no_grad():
        for j, data_val in enumerate(valid_loader):
            input_val, label_val = data_val
            input_val = Variable(input_val[:, None, :]).float()
            label_val = Variable(label_val).long()

            if torch.cuda.is_available():
                input_val = input_val.cuda()
                label_val = label_val.cuda()

            out2 = net(input_val)
            loss_val = criterion(out2, label_val).cpu()
            loss_item2 += loss_val.data.item()
            _, predicted = torch.max(out2.data, 1)

            for k in range(4):
                predic_count[k] += list(predicted).count(k)
                sampl_count[k] += list(label_val).count(k)
            for k in range(len(predicted)):
                if predicted[k] == label_val[k]:
                    correct_count[predicted[k]] += 1

            total += int(label_val.size(0))
            correct += int((predicted == label_val).sum())

    loss_val_list.append(loss_val)
    accuracy = round(100 * float(correct / total), 6)
    accuracy_list.append(accuracy)

    for i in range(4):
        if predic_count[i] == 0:
            precision = 0
        else:
            precision = round(100 * float(correct_count[i] / predic_count[i]), 6)
        if sampl_count[i] == 0:
            recall = 0
        else:
            recall = round(100 * float(correct_count[i] / sampl_count[i]), 6)
        precision_list[i].append(precision)
        recall_list[i].append(recall)
        if precision == 0 and recall == 0:
            F1_score += 0
        else:
            F1_score += (2 * precision * recall / (precision + recall)) / 4
    Macro_F1_list.append(F1_score)
    if accur_temp < 0:
        accur_temp = accuracy
        torch.save(net.state_dict(), os.getcwd() + '/outputs' + '/' + order + '/' + 'best_ckpt.pth')
    else:
        if accuracy > accur_temp:
            accur_temp = accuracy
            torch.save(net.state_dict(), os.getcwd() + '/outputs' + '/' + order + '/' + 'best_ckpt.pth')

    log = 'Accuracy of epoch ' + str(epoch + 1) + ': ' + str(accuracy)
    print(local_time() + log + log_time)
    file.write(local_time() + log + log_time + '\n')
    file.flush()




torch.save(net.state_dict(), output_dir + '/' + 'last_ckpt.pth')
np.save(output_dir + '/' + 'precision_list', np.array(precision_list))
np.save(output_dir + '/' + 'recall_list', np.array(recall_list))
np.save(output_dir + '/' + 'accuracy_list', np.array(accuracy_list))
np.save(output_dir + '/' + 'Macro_F1_list', np.array(Macro_F1_list))
np.save(output_dir + '/' + 'loss_train_list', np.array(loss_train_list))
np.save(output_dir + '/' + 'loss_val_list', np.array(loss_val_list))

print('Final accuracy:', accuracy_list[-1])
print('pr=' + str((precision_list[0][-1] + precision_list[1][-1] + precision_list[2][-1]) / 4))
print('re=' + str((recall_list[0][-1] + recall_list[1][-1] + recall_list[2][-1]) / 4))
print('ma=' + str(Macro_F1_list[-1]))
























