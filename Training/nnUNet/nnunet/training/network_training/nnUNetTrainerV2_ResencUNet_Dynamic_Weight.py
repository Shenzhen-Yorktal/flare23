import torch
import numpy as np
from torch.cuda.amp import autocast
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from nnunet.training.loss_functions.dice_loss import Dynamic_WeightedSoftDiceLoss, Dynamic_Weighted_DC_and_CE_loss
from nnunet.training.loss_functions.dice_loss import ClassWeightedMultipleOutputLoss, DC_and_Focal_Loss
from nnunet.training.network_training.nnUNet_variants.architectural_variants.nnUNetTrainerV2_ResencUNet import \
    nnUNetTrainerV2_ResencUNet
from nnunet.training.dataloading.dataset_loading import DataLoader2D, DataLoader3D
from nnunet.training.dataloading.dataset_loading_class_first import DataLoader3DMultiMaskData


class nnUNetTrainerV2_ResencUNet_Dynamic_Weight(nnUNetTrainerV2_ResencUNet):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.do_bg = False
        self.do_mean = True

    def initialize(self, training=True, force_load_plans=False):
        super().initialize(training=training, force_load_plans=force_load_plans)
        soft_dice_kwargs = {'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': self.do_bg}
        alpha = None
        # alpha = torch.Tensor([0.25] * 14 + [0.75])
        focal_kwargs = {'alpha': alpha, 'gamma': 2}
        self.loss = DC_and_Focal_Loss(soft_dice_kwargs, focal_kwargs)
        self.loss = ClassWeightedMultipleOutputLoss(self.loss, self.ds_loss_weights)

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """
        gradient clipping improves training stability

        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']
        properties = data_dict['properties']
        weight = self.get_weight_2(properties)

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        self.optimizer.zero_grad()

        if self.fp16:
            with autocast():
                output = self.network(data)
                del data
                l = self.loss(output, target, weight)

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            output = self.network(data)
            del data
            l = self.loss(output, target, weight)

            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target

        return l.detach().cpu().numpy()

    def get_weight(self, properties):
        shape_weight = (len(properties), len(properties[0]['class_locations']) + 1)
        classes_list = [prop['classes'].astype(np.int16) for prop in properties]

        weight = torch.zeros(shape_weight)
        t = -1 if self.do_bg else 0
        for i in range(shape_weight[0]):
            classes = classes_list[i][classes_list[i] > t]
            weight[i][classes] = 1
        return weight

    def get_weight_2(self, properties):
        # shape_weight = (len(properties), len(properties[0]['class_locations']) + 1)
        # weight = torch.ones(shape_weight)
        # for i in range(shape_weight[0]):
        #    weight[i][14] = 2
        weight = torch.ones(15)
        weight[14] = 2
        return weight

    def get_basic_generators(self):
        self.load_dataset()
        self.do_split()

        if self.threeD:
            dl_tr = DataLoader3DMultiMaskData(self.dataset_tr, self.basic_generator_patch_size, self.patch_size,
                                              self.batch_size,
                                              False, oversample_foreground_percent=self.oversample_foreground_percent,
                                              pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
            dl_val = DataLoader3DMultiMaskData(self.dataset_val, self.patch_size, self.patch_size, self.batch_size,
                                               False,
                                               oversample_foreground_percent=self.oversample_foreground_percent,
                                               pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
        else:
            dl_tr = DataLoader2D(self.dataset_tr, self.basic_generator_patch_size, self.patch_size, self.batch_size,
                                 oversample_foreground_percent=self.oversample_foreground_percent,
                                 pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
            dl_val = DataLoader2D(self.dataset_val, self.patch_size, self.patch_size, self.batch_size,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
        return dl_tr, dl_val
