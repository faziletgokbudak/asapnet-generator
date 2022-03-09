"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import sys

import cv2

import data
from models.networks import ASAPNetsGenerator
from options.test_options import TestOptions
import torch

# parse options
from util.util import calculate_psnr, calculate_ssim

opt = TestOptions().parse()


# print options to help debugging
print(' '.join(sys.argv))
dataloader = data.create_dataloader(opt)

model_path = opt.model_path + '/' + opt.filter + '.pth'

generator = ASAPNetsGenerator(opt).double()

generator.load_state_dict(torch.load(model_path))
# generator.load_state_dict(model_path)
# generator = torch.load(model_path)
# generator.eval()

with torch.no_grad():
    for i, data_i in enumerate(dataloader, 0):
        out_data, _ = generator(data_i['image'])

        pred = torch.squeeze(out_data, 0)
        # pred = torch.squeeze(target, 0)
        pred = torch.swapaxes(pred, 0, 2)
        pred = torch.swapaxes(pred, 0, 1)

        groundtruth = torch.squeeze(data_i['label'], 0)
        # pred = torch.squeeze(target, 0)
        groundtruth = torch.swapaxes(groundtruth, 0, 2)
        groundtruth = torch.swapaxes(groundtruth, 0, 1)

        pred_np = pred.cpu().detach().numpy() * 255.
        groundtruth_np = groundtruth.cpu().detach().numpy() * 255.

        print('aaa')
        # bgr = cv2.cvtColor(np.array([pred_np, test_ycbcr[:, :, 1],
        #                              test_ycbcr[:, :, 2]]).transpose([1, 2, 0]), cv2.COLOR_YCrCb2BGR)

        # cv2.imwrite('/Users/faziletgokbudak/PycharmProjects/ASAPNet-tensorflow/model/label_y_LLF_a2_s02.png', groundtruth_np)
        cv2.imwrite('/Users/faziletgokbudak/PycharmProjects/ASAPNet-tensorflow/asapnet_outputs/asapnet_UM_overfit.png', pred_np)

        # save_image(out_data, '/Users/faziletgokbudak/PycharmProjects/ASAPNet-tensorflow/model/out_y.png')

        # plt.figure()
        # plt.imshow(pred)
        # plt.show()
