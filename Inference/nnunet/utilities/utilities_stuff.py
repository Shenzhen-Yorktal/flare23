#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import torch
import os
import numpy as np
import cc3d  # connected-components-3d
import fastremap

from torch import nn


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, input):
        return input


class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        super(Upsample, self).__init__()
        self.align_corners = align_corners
        self.mode = mode
        self.scale_factor = scale_factor
        self.size = size

    def forward(self, x):
        return nn.functional.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode,
                                         align_corners=self.align_corners)


def subfiles(folder, suffix=None):
    res = [i for i in os.listdir(folder) if os.path.isfile(os.path.join(folder, i))
            and (suffix is None or i.endswith(suffix))]
    return res

   
def resample_patient(data, original_spacing, target_spacing):
    """
    :param data:
    :param original_spacing:
    :param target_spacing:
    then resample along lowres axis with order_z_data/order_z_seg instead of order_data/order_seg

    :return:
    """
    shape = np.array(data[0].shape)
    new_shape = np.round(((np.array(original_spacing) / np.array(target_spacing)).astype(float) * shape)).astype(int)

    data_torch = torch.from_numpy(data).to(torch.float32)
    data_torch = torch.unsqueeze(data_torch, 0)
    new_size = new_shape.tolist()
    reshaped_final_data = nn.functional.interpolate(data_torch, size=new_size, mode='trilinear', align_corners=False)
    reshaped_final_data = torch.squeeze(reshaped_final_data, 0)
    reshaped_final_data = reshaped_final_data.numpy()

    return reshaped_final_data


def keep_large_connected_object(seg,label):
    labels_in = seg == label
    labels_out = cc3d.connected_components(labels_in, connectivity=26)
    areas = {}
    for label, extracted in cc3d.each(labels_out, binary=True, in_place=True):
        areas[label] = fastremap.foreground(extracted)
    candidates = sorted(areas.items(), key=lambda item: item[1], reverse=True)
    unvalid = [i[0] for i in candidates[1:] if i[1] < 0.1 * candidates[0][1]]
    seg_map = np.in1d(labels_out, unvalid).reshape(labels_in.shape)
    return seg_map


def keep_largest_connected_object(seg,label):
    labels_in = seg == label
    labels_out = cc3d.connected_components(labels_in, connectivity=26)
    areas = {}
    for label, extracted in cc3d.each(labels_out, binary=True, in_place=True):
        areas[label] = fastremap.foreground(extracted)
    candidates = sorted(areas.items(), key=lambda item: item[1], reverse=True)
    unvalid = [i[0] for i in candidates[1:]]
    seg_map = np.in1d(labels_out, unvalid).reshape(labels_in.shape)
    return seg_map
    

def keep_largest_connected_area(seg, roi=False):
    seg_map = np.ones_like(seg)

    if roi:
        largest_list = []
        large_list = [1]
    else:
        largest_list = [1, 3, 4, 7, 8, 11]
        large_list = [2, 9, 13]
    for i in largest_list:
        seg_map -= keep_largest_connected_object(seg, i)
    for i in large_list:
        seg_map -= keep_large_connected_object(seg, i)
    return seg * seg_map


def slice_argmax(class_probabilities):
    step = class_probabilities.shape[1] // 100 + 1
    C, Z, X, Y = class_probabilities.shape
    result = np.zeros((Z, X, Y))
    z = class_probabilities.shape[1]
    stride = int(z / step)
    step1 = [i * stride for i in range(step)] + [z]
    for i in range(step):
        result[step1[i]:step1[i + 1]] = torch.argmax(class_probabilities[:, step1[i]:step1[i + 1]].cuda(), 0).cpu().numpy()
    torch.cuda.empty_cache()
    return result
