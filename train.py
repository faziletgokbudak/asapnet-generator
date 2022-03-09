"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import sys
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from torch.optim import Adam
from torchvision.utils import save_image

import data
from options.train_options import TrainOptions
from models.networks.generator import ASAPNetsGenerator
import torch

# parse options
from util.util import calculate_psnr, calculate_ssim

opt = TrainOptions().parse()


# print options to help debugging
print(' '.join(sys.argv))


# create trainer for our model
generator = ASAPNetsGenerator(opt).double()

# create tool for counting iterations

optimizer = Adam(generator.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
criterion = torch.nn.L1Loss()

dataloader = data.create_dataloader(opt)
# load the dataset

loss_list = []
for epoch in range(opt.epoch):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data_i in enumerate(dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        # img = data_i['image']

        output, _ = generator(data_i['image'])

        loss = criterion(output, data_i['label'])
        save_image(data_i['image'], '/Users/faziletgokbudak/PycharmProjects/ASAPNet-tensorflow/model/in_y.png')
        save_image(data_i['label'], '/Users/faziletgokbudak/PycharmProjects/ASAPNet-tensorflow/model/label_UM_y.png')

        my_loss = torch.sum(torch.abs(output - data_i['label']))
        loss_list.append(loss)
        print('loss:', loss)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

# plt.figure()
# # plt.plot(loss_list, label='MSE loss')
# plt.plot(loss_list)
# # plt.legend(loc='lower right')z
# # plt.title('PSNR Plot')
# plt.xlabel('epoch')
# plt.ylabel('L1 loss')
# plt.show()

if not os.path.exists(opt.model_path):
    os.makedirs(opt.model_path)

model_path = opt.model_path + '/' + opt.filter + '.pth'

torch.save(generator.state_dict(), model_path)

print('Finished Training')
print('Training was successfully finished.')

