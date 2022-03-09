"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import cv2

from data.base_dataset import BaseDataset, get_params, get_transform
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import util.util as util
import os
import torch
from util.util import crop_center


class Pix2pixDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--no_pairing_check', action='store_true',
                            help='If specified, skip sanity check of correct label-image file pairing')
        return parser

    def initialize(self, opt):
        self.opt = opt

        label_paths, image_paths = self.get_paths(opt)

        label_paths = label_paths[:opt.max_dataset_size]
        image_paths = image_paths[:opt.max_dataset_size]

        self.label_paths = label_paths
        self.image_paths = image_paths

        size = len(self.label_paths)
        self.dataset_size = size

    def get_paths(self, opt):
        label_paths = []
        image_paths = []
        assert False, "A subclass of Pix2pixDataset must override self.get_paths(self, opt)"
        return label_paths, image_paths

    def paths_match(self, path1, path2):
        filename1_without_ext = os.path.splitext(os.path.basename(path1))[0]
        filename2_without_ext = os.path.splitext(os.path.basename(path2))[0]
        return filename1_without_ext == filename2_without_ext

    def __getitem__(self, index):
        # Label Image
        label_path = self.label_paths[index]
        # label = Image.open(label_path)
        label = cv2.imread(label_path)#[:, :, 0] / 255.
        label_y = cv2.cvtColor(label, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
        # label_y = crop_center(label_y, self.opt.center_crop_size, self.opt.center_crop_size)
        label_y = cv2.resize(label_y, (self.opt.resize_size, self.opt.resize_size),
                             interpolation=cv2.INTER_CUBIC) / 255.
        label_tensor = torch.unsqueeze(torch.from_numpy(label_y), 0)

        # #label.save('%s.jpg'%label_path[:-4],'JPEG')
        # params = get_params(self.opt, label.size)
        # transform_label = get_transform(self.opt, params)
        # # transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        # # # label = label.convert('RGB')
        # transform_image = get_transform(self.opt, params)
        # label_tensor = transform_image(label)

        '''
        print(label_tensor.shape)
        target = label_tensor[:, 280, 100]
        print(target)
        source = label_tensor[:, 310, 70]
        print(source)
        label_tensor[:, 272:333, 99:122] = 2
        label_tensor[:, 272-30:333-30, 99:122] = 3
        '''
        # label_tensor[label_tensor == 255] = self.opt.label_nc  # 'unknown' is opt.label_nc
        # assert ((torch.max(label_tensor)+1) == self.opt.label_nc, \
        #         "uncorrect number of labels (--label_nc=%f does not match the given labels=%f)"  % \
        #         (self.opt.label_nc,torch.max(label_tensor)+1))

            # input image (real images)
        image_path = self.image_paths[index]
        # image = Image.open(image_path)
        image = cv2.imread(image_path)#[:, :, 0] / 255.
        image_y = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
        # image_y = crop_center(image_y, self.opt.center_crop_size, self.opt.center_crop_size)
        image_y = cv2.resize(image_y, (self.opt.resize_size, self.opt.resize_size),
                             interpolation=cv2.INTER_CUBIC) / 255.
        # print(image)
        image_tensor = torch.unsqueeze(torch.from_numpy(image_y), 0)

        #
        # image = torch.from_numpy(cv2.imread(image_path))
        #image.save('%s.jpg'%image_path[:-4],'JPEG')
        # image = image.convert('RGB')

        # transform_image = get_transform(self.opt, params)
        # image_tensor = transform_image(image)

        input_dict = {'label': label_tensor,
                      'image': image_tensor,
                      'path': image_path,
                      }

        # Give subclasses a chance to modify the final output
        self.postprocess(input_dict)

        return input_dict

    def postprocess(self, input_dict):
        return input_dict

    def __len__(self):
        return 1#self.dataset_size
