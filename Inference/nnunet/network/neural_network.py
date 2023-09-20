import torch
import numpy as np

from torch import nn
from torch.cuda.amp import autocast
from scipy.ndimage.filters import gaussian_filter


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

    def get_device(self):
        if next(self.parameters()).device == "cpu":
            return "cpu"
        else:
            return next(self.parameters()).device.index

    def set_device(self, device):
        if device == "cpu":
            self.cpu()
        else:
            self.cuda(device)

    def forward(self, x):
        raise NotImplementedError


class SegmentationNetwork(NeuralNetwork):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.input_shape_must_be_divisible_by = None
        self.conv_op = None
        self.num_classes = None
        self._gaussian_3d = self._patch_size_for_gaussian_3d = None
        self._gaussian_2d = self._patch_size_for_gaussian_2d = None

    def predict_3D(self, x, step_size=0.5, sliding=True, patch_size=None):
        torch.cuda.empty_cache()
        with autocast():
            with torch.no_grad():
                if sliding:
                    res = self._internal_predict_3D_3Dconv_tiled(x, step_size, patch_size)
                else:
                    res = self._internal_predict_3D_3Dconv(x, step_size, patch_size)

        return res

    @staticmethod
    def _get_gaussian(patch_size, sigma_scale=1. / 8):
        tmp = np.zeros(patch_size)
        center_coords = [i // 2 for i in patch_size]
        sigmas = [i * sigma_scale for i in patch_size]
        tmp[tuple(center_coords)] = 1
        gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
        gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
        gaussian_importance_map = gaussian_importance_map.astype(np.float32)

        gaussian_importance_map[gaussian_importance_map == 0] = np.min(
            gaussian_importance_map[gaussian_importance_map != 0])

        return gaussian_importance_map

    @staticmethod
    def _compute_steps_for_sliding_window(patch_size, image_size, step_size):
        target_step_sizes_in_voxels = [i * step_size for i in patch_size]

        num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in
                     zip(image_size, target_step_sizes_in_voxels, patch_size)]

        steps = []
        for dim in range(len(patch_size)):
            max_step_value = image_size[dim] - patch_size[dim]
            if num_steps[dim] > 1:
                actual_step_size = max_step_value / (num_steps[dim] - 1)
            else:
                actual_step_size = 99999999999

            steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]

            steps.append(steps_here)

        return steps

    def _internal_predict_3D_3Dconv_tiled(self, x, step_size, patch_size):
        raise NotImplementedError

    def _internal_predict_3D_3Dconv(self, x, step_size, patch_size):
        raise NotImplementedError

    def _internal_maybe_mirror_and_pred_3D(self, x, mult=None):
        result_torch = self.inference_apply_nonlin(self(x))

        if mult is not None:
            result_torch[:, :] *= mult

        return result_torch

    @staticmethod
    def _pad_nd_image(image, new_shape=None):
        old_shape = np.array(image.shape[-len(new_shape):])
        num_axes_nopad = len(image.shape) - len(new_shape)
        new_shape = [max(new_shape[i], old_shape[i]) for i in range(len(new_shape))]

        difference = new_shape - old_shape
        pad_below = difference // 2
        pad_above = difference // 2 + difference % 2
        pad_list = [[0, 0]] * num_axes_nopad + list([list(i) for i in zip(pad_below, pad_above)])

        if not ((all([i == 0 for i in pad_below])) and (all([i == 0 for i in pad_above]))):
            res = np.pad(image, pad_list, "constant", constant_values=0.0)
        else:
            res = image

        pad_list = np.array(pad_list)
        pad_list[:, 1] = np.array(res.shape) - pad_list[:, 1]
        slicer = list(slice(*i) for i in pad_list)
        return res, slicer
        
        
    def inference_apply_nonlin(self, x):
        return nn.functional.softmax(x, 1)
        
