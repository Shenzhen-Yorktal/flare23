import argparse
import torch
import os

import numpy as np
import skimage.measure as sm
import torch.nn.functional as f

from threading import Thread
from SimpleITK import WriteImage, ReadImage, GetArrayFromImage, GetImageFromArray
from nnunet.network.FabiansUNet import FabiansUNet
from nnunet.network.BaseUNet import Generic_UNet
from nnunet.network.RepVGGFast import VGGFastUNet
from nnunet.utilities.utilities_stuff import subfiles, resample_patient, keep_largest_connected_area, slice_argmax

net_pars_fine = {'mean': 83.560104, 'std': 136.6215, 'lower_bound1': -967.0,
                 'upper_bound1': 291.0, 'patch_size': np.array([32, 128, 192]),
                 'target_spacing': np.array([4.0, 1.2, 1.2])}
net_pars_fine2 = {'mean': 79.64796, 'std': 139.33594, 'lower_bound1': -965.0,
                  'upper_bound1': 276.0, 'patch_size': np.array([64, 128, 224]),
                  'target_spacing': np.array([2.5, 0.81835938, 0.81835938])}

ROI_SLIDING_THRESHOLD = 240
ROI_THICKNESS_THRESHOLD = 400
FINE_SIZE_THRESHOLD = 400


def get_data(**kwargs):
    data_files = kwargs["data_files"]
    data_itk = [ReadImage(f) for f in data_files]
    data = np.vstack([GetArrayFromImage(d)[None] for d in data_itk])
    direc = data_itk[0].GetDirection()
    if direc[-1] < 0:
        data = data[:, ::-1]
    data = data.astype(np.float32)

    original_spacing = np.array(data_itk[0].GetSpacing())[[2, 1, 0]]
    original_size = data[0].shape
    crop_box = [[0, data.shape[1] + 1], [0, data.shape[2] + 1], [0, data.shape[3] + 1]]

    itk_origin = data_itk[0].GetOrigin()
    itk_spacing = data_itk[0].GetSpacing()
    itk_direction = data_itk[0].GetDirection()

    step = [int(np.round(5.0 / original_spacing[0])), 4, 4]
    data_r = data[:, ::step[0], ::step[1], ::step[2]].copy()

    data_r = np.clip(data_r, -1024, 1024)
    data_r = (data_r - data_r.mean()) / (data_r.std() + 1e-8)

    return data, data_r, itk_direction, itk_spacing, itk_origin, crop_box, original_size, original_spacing, direc


def get_model_roi(**kwargs):
    with torch.no_grad():
        model_roi = kwargs["model_roi"]
        net_roi = torch.load(model_roi)
        net_roi.cuda()
        net_roi.eval()
        dumpy_data = np.zeros((1, 16, 16, 16), dtype=np.float32)
        _ = net_roi.predict_3D(dumpy_data, step_size=1, sliding=False, patch_size=None)
    return net_roi


def get_model_fine(**kwargs):
    with torch.no_grad():
        model_fine = kwargs["model_fine"]
        net_fine = torch.load(model_fine)
        net_fine.cuda()
        net_fine.eval()
    return net_fine
    
    
def get_model_fine2(**kwargs):
    with torch.no_grad():
        model_fine = kwargs["model_fine"]
        net_fine = torch.load(model_fine)
        net_fine.cuda()
        net_fine.eval()
        
        data = kwargs["data"]
        data = resample_patient(data, original_spacing, net_pars_fine2['target_spacing'])

        mean_intensity2 = net_pars_fine2['mean']
        std_intensity2 = net_pars_fine2['std']
        lower_bound2 = net_pars_fine2['lower_bound1']
        upper_bound2 = net_pars_fine2['upper_bound1']
        data = np.clip(data, lower_bound2, upper_bound2)
        data = (data - mean_intensity2) / std_intensity2
    
    return net_fine, data


class MyThread(Thread):
    def __init__(self, *args, **kwargs):
        super(MyThread, self).__init__(*args, **kwargs)
        self.result = None

    def run(self) -> None:
        self.result = self._target(**self._kwargs)

    def get_result(self):
        return self.result


def get_longest_sequence(x):
    x = x.reshape(-1)
    l = 1
    res_start = x[0]
    res_end = x[-1] + 1
    temp_start = x[0]
    for i in range(1, len(x)):
        if x[i] - x[i - 1] != 1:
            temp_len = x[i] - temp_start + 1
            if temp_len > l:
                l = temp_len
                res_start = temp_start
                res_end = x[i - 1]+1
                temp_start = x[i]
        if i == len(x) - 1:
            temp_len = x[i] - temp_start + 1
            if temp_len > l:
                res_start = temp_start
                res_end = x[i]+1
    return res_start, res_end


if __name__ == "__main__":
    """ We predict roi in fastest mode for saving time. The step-size is 1 in roi extracting"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", '--input_folder', help="Must contain all modalities for each patient in the correct"
                                                     " order (same as training). Files must be named "
                                                     "CASENAME_XXXX.nii.gz where XXXX is the modality "
                                                     "identifier (0000, 0001, etc)", required=True)
    parser.add_argument('-o', "--output_folder", required=True, help="folder for saving predictions")

    args = parser.parse_args()
    input_folder = args.input_folder
    output_folder = args.output_folder

    model_roi = r"/models/model_roi.pt"
    model_fine = r"/models/model_fine.pt"
    model_fine2 = r"/models/model_fine2.pt"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = subfiles(input_folder, suffix=".nii.gz")
    case_ids = np.unique([f[:-12] for f in files])

    output_files = [os.path.join(output_folder, cid + ".nii.gz") for cid in case_ids]
    list_of_lists = [[os.path.join(input_folder, f) for f in files if f[:len(j)].startswith(j) and
                      len(f) == (len(j) + 12)] for j in case_ids]

    torch.cuda.empty_cache()
    for idx in range(len(list_of_lists)):
        data_files = list_of_lists[idx]
        task1 = MyThread(target=get_data, kwargs={"data_files": data_files})
        task2 = MyThread(target=get_model_roi, kwargs={"model_roi": model_roi})
        task3 = MyThread(target=get_model_fine, kwargs={"model_fine": model_fine})
        task1.start()
        task2.start()
        task3.start()
        task1.join()
        task2.join()
        with torch.no_grad():
            data, data_r, itk_direction, itk_spacing, itk_origin, crop_box, original_size, original_spacing, direc = task1.get_result()
            net_roi = task2.get_result()
            net_roi.cuda()
            net_roi.eval()
            if data_r.shape[1] > ROI_SLIDING_THRESHOLD:
                seg = net_roi.predict_3D(data_r, step_size=1, patch_size=[80, 128, 128])
            else:
                seg = net_roi.predict_3D(data_r, step_size=1, sliding=False, patch_size=None)
        del data_r, net_roi, task2
        torch.cuda.empty_cache()
        
        current_shape = seg.shape
        reg = sm.regionprops(seg)

        if len(reg) == 0:
            print('roi result is none')
            seg = np.zeros(original_size, dtype=np.int8)
            seg_resized_itk = GetImageFromArray(seg)
            seg_resized_itk.SetSpacing(itk_spacing)
            seg_resized_itk.SetOrigin(itk_origin)
            seg_resized_itk.SetDirection(itk_direction)
            WriteImage(seg_resized_itk, output_files[idx])
            continue

        roi_box = list(reg[0].bbox)
        roi_box2 = [0, 0, 0, 0, 0, 0]
        sum_z = np.sum(seg, axis=(1, 2))
        sum_y = np.sum(seg, axis=(0, 2))
        sum_x = np.sum(seg, axis=(0, 1))
        pos_z = np.argwhere(sum_z > 0)
        pos_y = np.argwhere(sum_y > 0)
        pos_x = np.argwhere(sum_x > 0)
        res_z = get_longest_sequence(pos_z)
        roi_box2[0] = res_z[0]
        roi_box2[3] = res_z[-1]
        res_y = get_longest_sequence(pos_y)
        roi_box2[1] = res_y[0]
        roi_box2[4] = res_y[-1]
        res_x = get_longest_sequence(pos_x)
        roi_box2[2] = res_x[0]
        roi_box2[5] = res_x[-1]
        for i in range(6):
            if roi_box[i] != roi_box2[i]:
                seg_itk = GetImageFromArray(seg.astype(np.uint8))
                WriteImage(seg_itk, output_files[idx])
                print(output_files[idx], roi_box)
                print(output_files[idx], roi_box2)
                break
        continue

        roi_box_lower = np.array(roi_box[:3])
        roi_box_upper = np.array(roi_box[3:])
        roi_ratio = np.array(original_size) / np.array(current_shape)
        roi_box_lower = [int(lb) for lb in np.floor(roi_box_lower * roi_ratio)]
        roi_box_upper = [int(ub) for ub in np.ceil(roi_box_upper * roi_ratio)]

        del seg
        padding_length_upper1 = [20, 5, 5]  # [20, 15, 15]
        padding_length_lower1 = [40, 5, 5]  # [140, 15, 15]
        padding_crop_upper = [int(np.round(i / j)) for i, j in zip(padding_length_upper1, original_spacing)]
        padding_crop_lower = [int(np.round(i / j)) for i, j in zip(padding_length_lower1, original_spacing)]

        for c in range(3):
            crop_box[c][1] = np.min((crop_box[c][0] + roi_box_upper[c] + padding_crop_upper[c], original_size[c]))
            crop_box[c][0] = np.max((0, crop_box[c][0] + roi_box_lower[c] - padding_crop_lower[c]))

        data = data[:, crop_box[0][0]:crop_box[0][1], crop_box[1][0]:crop_box[1][1], crop_box[2][0]:crop_box[2][1]]
        size_ac = list(data.shape[1:])

        task4 = MyThread(target=get_model_fine2, kwargs={"model_fine": model_fine2, "data": data})
        task4.start()
        
        data1 = resample_patient(data, original_spacing, net_pars_fine['target_spacing'])

        mean_intensity = net_pars_fine['mean']
        std_intensity = net_pars_fine['std']
        lower_bound = net_pars_fine['lower_bound1']
        upper_bound = net_pars_fine['upper_bound1']
        data1 = np.clip(data1, lower_bound, upper_bound)
        data1 = (data1 - mean_intensity) / std_intensity

        task3.join()
        with torch.no_grad():
            net_fine = task3.get_result()
            softmax = net_fine.predict_3D(data1, step_size=0.5, patch_size=net_pars_fine['patch_size'])[None][0]

            del data1, net_fine, task3
            torch.cuda.empty_cache()

            current_shape = softmax.shape
            if size_ac[0] > FINE_SIZE_THRESHOLD:
                softmax = softmax.detach().cpu()
                torch.cuda.empty_cache()
                if np.any([i != j for i, j in zip(np.array(current_shape[1:]), np.array(size_ac))]):
                    step = size_ac[0] // 150 + 1
                    seg_old_spacing = np.zeros(size_ac)
                    z = current_shape[1]
                    stride = int(z / step)
                    step1 = [i * stride for i in range(step)] + [z]
                    z = size_ac[0]
                    stride = int(z / step)
                    step2 = [i * stride for i in range(step)] + [z]
                    for i in range(step):
                        size = list(size_ac)
                        size[0] = step2[i + 1] - step2[i]
                        slicer = softmax[:, step1[i]:step1[i + 1]][None].half()
                        slicer = f.interpolate(slicer.cuda(), mode='trilinear', size=size, align_corners=True)[0]
                        seg_old_spacing[step2[i]:step2[i + 1]] = slice_argmax(slicer)
                        del slicer
                        torch.cuda.empty_cache()
                else:
                    seg_old_spacing = slice_argmax(softmax)
            else:
                if np.any([i != j for i, j in zip(np.array(current_shape[1:]), np.array(size_ac))]):
                    softmax = f.interpolate(softmax[None].half(), mode='trilinear', size=size_ac, align_corners=True)[0]
                    seg_old_spacing = torch.argmax(softmax, 0).cpu().numpy()
                else:
                    seg_old_spacing = torch.argmax(softmax, 0).cpu().numpy()
            del softmax
            torch.cuda.empty_cache()

            task4.join()
            net_fine2, data = task4.get_result()
            print(data.shape)
            softmax = net_fine2.predict_3D(data, step_size=0.8, patch_size=net_pars_fine2['patch_size'])[None][0]
            del data, net_fine2, task4
            torch.cuda.empty_cache()

            current_shape = softmax.shape
            print('p_size', size_ac[0] * size_ac[1] * size_ac[2])
            if size_ac[0] > FINE_SIZE_THRESHOLD:
                softmax = softmax.detach().cpu()
                torch.cuda.empty_cache()
                if np.any([i != j for i, j in zip(np.array(current_shape[1:]), np.array(size_ac))]):
                    step = size_ac[0] // 150 + 1
                    seg_old_spacing2 = np.zeros(size_ac)
                    z = current_shape[1]
                    stride = int(z / step)
                    step1 = [i * stride for i in range(step)] + [z]
                    z = size_ac[0]
                    stride = int(z / step)
                    step2 = [i * stride for i in range(step)] + [z]
                    for i in range(step):
                        size = size_ac
                        size[0] = step2[i + 1] - step2[i]
                        slicer = softmax[:, step1[i]:step1[i + 1]][None].half()
                        slicer = f.interpolate(slicer.cuda(), mode='trilinear', size=size, align_corners=True)[0]
                        seg_old_spacing2[step2[i]:step2[i + 1]] = slice_argmax(slicer)
                        del slicer
                        torch.cuda.empty_cache()
                else:
                    seg_old_spacing2 = slice_argmax(softmax)
            else:
                if np.any([i != j for i, j in zip(np.array(current_shape[1:]), np.array(size_ac))]):
                    softmax = f.interpolate(softmax[None].half(), mode='trilinear', size=size_ac, align_corners=True)[0]
                    seg_old_spacing2 = torch.argmax(softmax, 0).cpu().numpy()
                else:
                    seg_old_spacing2 = torch.argmax(softmax, 0).cpu().numpy()
            del softmax

            seg_old_spacing2[seg_old_spacing2 > 13] = seg_old_spacing[seg_old_spacing2 > 13]
            seg_old_spacing2[seg_old_spacing > 13] = 14
            del seg_old_spacing

            if size_ac[0] < FINE_SIZE_THRESHOLD:
                seg_old_spacing2 = keep_largest_connected_area(seg_old_spacing2.astype(np.uint8))

            seg_old_size = np.zeros(original_size)
            for c in range(3):
                crop_box[c][1] = np.min((crop_box[c][0] + seg_old_spacing2.shape[c], original_size[c]))
            seg_old_size[crop_box[0][0]:crop_box[0][1], crop_box[1][0]:crop_box[1][1], crop_box[2][0]:crop_box[2][1]] = \
                seg_old_spacing2
            del seg_old_spacing2

            if direc[-1] < 0:
                seg_old_size = seg_old_size[::-1]
            seg_resized_itk = GetImageFromArray(seg_old_size.astype(np.uint8))
            seg_resized_itk.SetSpacing(itk_spacing)
            seg_resized_itk.SetOrigin(itk_origin)
            seg_resized_itk.SetDirection(itk_direction)
            WriteImage(seg_resized_itk, output_files[idx])

        torch.cuda.empty_cache()
