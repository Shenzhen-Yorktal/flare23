import torch
import numpy as np
from torch.cuda.amp import autocast
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from nnunet.training.loss_functions.dice_loss import ClassWeightedMultipleOutputLoss, DC_and_Focal_Loss
from nnunet.training.network_training.nnUNet_variants.architectural_variants.nnUNetTrainerV2_ResencUNet import nnUNetTrainerV2_ResencUNet
from nnunet.training.dataloading.dataset_loading import DataLoader2D, DataLoader3D
from nnunet.training.dataloading.dataset_loading_class_first import DataLoader3DMultiMaskData
from nnunet.training.network_training.nnUNetTrainerV2_ResencUNet_Dynamic_Weight import nnUNetTrainerV2_ResencUNet_Dynamic_Weight


class nnUNetTrainerV2_ResencUNet_finetuning(nnUNetTrainerV2_ResencUNet_Dynamic_Weight):

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.initial_lr = 1e-2
        self.probing = True
        self.max_num_epochs = 200
        self.num_val_batches_per_epoch = 2

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        for param in self.network.parameters():
            param.requires_grad = True
        # for param in self.network.decoder.deep_supervision_outputs.parameters():
        #     param.requires_grad = True
        self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.network.parameters()),
                                         self.initial_lr, weight_decay=self.weight_decay,
                                         momentum=0.95, nesterov=True)
        self.lr_scheduler = None


    def maybe_update_lr(self, epoch=None):
        if self.epoch < 50:
            # epoch 49 is max
            # we increase lr linearly from 0 to initial_lr
            lr = (self.epoch + 1) / 50 * self.initial_lr
            self.optimizer.param_groups[0]['lr'] = lr
            self.print_to_log_file("epoch:", self.epoch, "lr:", lr)
        else:
            if self.epoch < 100:
                for param in self.network.decoder.parameters():
                    param.requires_grad = True
                self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.network.parameters()),
                                                 self.initial_lr, weight_decay=self.weight_decay,
                                                 momentum=0.95, nesterov=True)
            else:
                for param in self.network.parameters():
                    param.requires_grad = True
                self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.network.parameters()),
                                                 self.initial_lr, weight_decay=self.weight_decay,
                                                 momentum=0.95, nesterov=True)

            if epoch is not None:
                ep = epoch - 49
            else:
                ep = self.epoch - 49
            assert ep > 0, "epoch must be >0"

            return super().maybe_update_lr(ep)


