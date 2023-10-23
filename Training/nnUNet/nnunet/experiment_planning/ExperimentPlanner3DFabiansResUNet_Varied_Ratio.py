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


from copy import deepcopy

import numpy as np
from nnunet.experiment_planning.experiment_planner_baseline_3DUNet_v21 import \
    ExperimentPlanner3D_v21
from nnunet.experiment_planning.common_utils import get_pool_and_conv_props
from nnunet.paths import *
from nnunet.network_architecture.generic_modular_residual_UNet import FabiansUNet


class ExperimentPlanner3DFabiansResUNet_Varied_Ratio(ExperimentPlanner3D_v21):
    """spacing ratio is about 5/spacing"""
    def __init__(self, folder_with_cropped_data, preprocessed_output_folder, only_lowres=False):
        super(ExperimentPlanner3DFabiansResUNet_Varied_Ratio, self).__init__(folder_with_cropped_data,
                                                                    preprocessed_output_folder, only_lowres)
        self.data_identifier = "nnUNetData_vary_ratio_plans_v2.1"  # "nnUNetData_FabiansResUNet_v2.1"
        self.plans_fname = join(self.preprocessed_output_folder,
                                "nnUNetPlans_FabiansResUNet_vr_plans_3D.pkl")
        self.fixed_spacing = np.array([[5, 5, 5], [1, 1, 1]]).astype(
            'float')
        self.preprocessor_name = "SpacingVaryRatioResamplePreprocessor3D"

    def get_target_size(self):
        """
        per default we use the 50th percentile=median for the target spacing. Higher spacing results in smaller data
        and thus faster and easier training. Smaller spacing results in larger data and thus longer and harder training

        For some datasets the median is not a good choice. Those are the datasets where the spacing is very anisotropic
        (for example ACDC with (10, 1.5, 1.5)). These datasets still have examples with a spacing of 5 or 6 mm in the low
        resolution axis. Choosing the median here will result in bad interpolation artifacts that can substantially
        impact performance (due to the low number of slices).
        """
        spacings = self.dataset_properties['all_spacings']
        spacings = np.vstack(spacings)
        sizes = self.dataset_properties['all_sizes']
        sizes = np.vstack(sizes)
        physical_size = sizes * spacings

        target = np.percentile(physical_size, self.target_spacing_percentile, 0)

        return target

    def get_properties_for_stage(self, current_spacing, original_spacing, original_shape, num_cases,
                                 num_modalities, num_classes):
        """
        We use FabiansUNet instead of Generic_UNet
        """

        new_median_shape = np.round(original_shape/np.array(current_spacing)).astype(int)
        # dataset_num_voxels = np.prod(new_median_shape) * num_cases
        dataset_num_voxels = np.prod(new_median_shape, dtype=np.int64) * num_cases

        # the next line is what we had before as a default. The patch size had the same aspect ratio as the median shape of a patient. We swapped t
        # input_patch_size = new_median_shape

        # compute how many voxels are one mm
        input_patch_size = 1 / np.array(current_spacing)

        # normalize voxels per mm
        input_patch_size /= input_patch_size.mean()

        # create an isotropic patch of size 512x512x512mm
        input_patch_size *= 1 / min(input_patch_size) * 512  # to get a starting value
        input_patch_size = np.round(input_patch_size).astype(int)

        temp_shape = np.array([96, 192, 192])
        # clip it to the median shape of the dataset because patches larger then that make not much sense
        input_patch_size = [min(i, j, k) for i, j, k in zip(input_patch_size, new_median_shape, temp_shape)]

        network_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, new_shp, \
        shape_must_be_divisible_by = get_pool_and_conv_props(current_spacing, input_patch_size,
                                                             self.unet_featuremap_min_edge_length,
                                                             self.unet_max_numpool)
        pool_op_kernel_sizes = [[1, 1, 1]] + pool_op_kernel_sizes
        blocks_per_stage_encoder = FabiansUNet.default_blocks_per_stage_encoder[:len(pool_op_kernel_sizes)]
        blocks_per_stage_decoder = FabiansUNet.default_blocks_per_stage_decoder[:len(pool_op_kernel_sizes) - 1]

        ref = FabiansUNet.use_this_for_3D_configuration
        here = FabiansUNet.compute_approx_vram_consumption(input_patch_size, self.unet_base_num_features,
                                                           self.unet_max_num_filters, num_modalities, num_classes,
                                                           pool_op_kernel_sizes, blocks_per_stage_encoder,
                                                           blocks_per_stage_decoder, 2, self.unet_min_batch_size, )
        while here > ref:
            axis_to_be_reduced = np.argsort(new_shp / new_median_shape)[-1]

            tmp = deepcopy(new_shp)
            tmp[axis_to_be_reduced] -= shape_must_be_divisible_by[axis_to_be_reduced]
            _, _, _, _, shape_must_be_divisible_by_new = \
                get_pool_and_conv_props(current_spacing, tmp,
                                        self.unet_featuremap_min_edge_length,
                                        self.unet_max_numpool,
                                        )
            new_shp[axis_to_be_reduced] -= shape_must_be_divisible_by_new[axis_to_be_reduced]

            # we have to recompute numpool now:
            network_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, new_shp, \
            shape_must_be_divisible_by = get_pool_and_conv_props(current_spacing, new_shp,
                                                                 self.unet_featuremap_min_edge_length,
                                                                 self.unet_max_numpool,
                                                                 )
            pool_op_kernel_sizes = [[1, 1, 1]] + pool_op_kernel_sizes
            blocks_per_stage_encoder = FabiansUNet.default_blocks_per_stage_encoder[:len(pool_op_kernel_sizes)]
            blocks_per_stage_decoder = FabiansUNet.default_blocks_per_stage_decoder[:len(pool_op_kernel_sizes) - 1]
            here = FabiansUNet.compute_approx_vram_consumption(new_shp, self.unet_base_num_features,
                                                               self.unet_max_num_filters, num_modalities, num_classes,
                                                               pool_op_kernel_sizes, blocks_per_stage_encoder,
                                                               blocks_per_stage_decoder, 2, self.unet_min_batch_size)
        input_patch_size = new_shp

        batch_size = FabiansUNet.default_min_batch_size
        batch_size = int(np.floor(max(ref / here, 1) * batch_size))

        # check if batch size is too large
        max_batch_size = np.round(self.batch_size_covers_max_percent_of_dataset * dataset_num_voxels /
                                  np.prod(input_patch_size, dtype=np.int64)).astype(int)
        max_batch_size = max(max_batch_size, self.unet_min_batch_size)
        batch_size = max(1, min(batch_size, max_batch_size))

        do_dummy_2D_data_aug = (max(input_patch_size) / input_patch_size[
            0]) > self.anisotropy_threshold

        plan = {
            'batch_size': batch_size,
            'num_pool_per_axis': network_num_pool_per_axis,
            'patch_size': input_patch_size,
            'median_patient_size_in_voxels': new_median_shape,
            'current_spacing': current_spacing,
            'original_spacing': original_spacing,
            'do_dummy_2D_data_aug': do_dummy_2D_data_aug,
            'pool_op_kernel_sizes': pool_op_kernel_sizes,
            'conv_kernel_sizes': conv_kernel_sizes,
            'num_blocks_encoder': blocks_per_stage_encoder,
            'num_blocks_decoder': blocks_per_stage_decoder
        }
        return plan

    def get_properties_for_stage_vary(self, current_spacing, original_spacing, original_shape, num_cases,
                                 num_modalities, num_classes):
        """
        We use FabiansUNet instead of Generic_UNet
        """

        new_median_shape = np.round(original_shape/np.array(current_spacing)).astype(int)
        # dataset_num_voxels = np.prod(new_median_shape) * num_cases
        dataset_num_voxels = np.prod(new_median_shape, dtype=np.int64) * num_cases

        # the next line is what we had before as a default. The patch size had the same aspect ratio as the median shape of a patient. We swapped t
        # input_patch_size = new_median_shape

        # compute how many voxels are one mm
        input_patch_size = 1 / np.array(current_spacing)

        # normalize voxels per mm
        input_patch_size /= input_patch_size.mean()

        # create an isotropic patch of size 512x512x512mm
        input_patch_size *= 1 / min(input_patch_size) * 512  # to get a starting value
        input_patch_size = np.round(input_patch_size).astype(int)

        temp_shape = np.array([96, 192, 192])
        # clip it to the median shape of the dataset because patches larger then that make not much sense
        input_patch_size = [min(i, j, k) for i, j, k in zip(input_patch_size, new_median_shape, temp_shape)]

        network_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, new_shp, \
        shape_must_be_divisible_by = get_pool_and_conv_props(current_spacing, input_patch_size,
                                                             self.unet_featuremap_min_edge_length,
                                                             self.unet_max_numpool)
        pool_op_kernel_sizes = [[1, 1, 1]] + pool_op_kernel_sizes
        blocks_per_stage_encoder = FabiansUNet.default_blocks_per_stage_encoder[:len(pool_op_kernel_sizes)]
        blocks_per_stage_decoder = FabiansUNet.default_blocks_per_stage_decoder[:len(pool_op_kernel_sizes) - 1]

        ref = FabiansUNet.use_this_for_3D_configuration
        here = FabiansUNet.compute_approx_vram_consumption(input_patch_size, self.unet_base_num_features,
                                                           self.unet_max_num_filters, num_modalities, num_classes,
                                                           pool_op_kernel_sizes, blocks_per_stage_encoder,
                                                           blocks_per_stage_decoder, 2, self.unet_min_batch_size, )
        while here > ref:
            axis_to_be_reduced = np.argsort(new_shp / new_median_shape)[-1]

            tmp = deepcopy(new_shp)
            tmp[axis_to_be_reduced] -= shape_must_be_divisible_by[axis_to_be_reduced]
            _, _, _, _, shape_must_be_divisible_by_new = \
                get_pool_and_conv_props(current_spacing, tmp,
                                        self.unet_featuremap_min_edge_length,
                                        self.unet_max_numpool,
                                        )
            new_shp[axis_to_be_reduced] -= shape_must_be_divisible_by_new[axis_to_be_reduced]

            # we have to recompute numpool now:
            network_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, new_shp, \
            shape_must_be_divisible_by = get_pool_and_conv_props(current_spacing, new_shp,
                                                                 self.unet_featuremap_min_edge_length,
                                                                 self.unet_max_numpool,
                                                                 )
            pool_op_kernel_sizes = [[1, 1, 1]] + pool_op_kernel_sizes
            blocks_per_stage_encoder = FabiansUNet.default_blocks_per_stage_encoder[:len(pool_op_kernel_sizes)]
            blocks_per_stage_decoder = FabiansUNet.default_blocks_per_stage_decoder[:len(pool_op_kernel_sizes) - 1]
            here = FabiansUNet.compute_approx_vram_consumption(new_shp, self.unet_base_num_features,
                                                               self.unet_max_num_filters, num_modalities, num_classes,
                                                               pool_op_kernel_sizes, blocks_per_stage_encoder,
                                                               blocks_per_stage_decoder, 2, self.unet_min_batch_size)
        input_patch_size = new_shp

        batch_size = FabiansUNet.default_min_batch_size
        batch_size = int(np.floor(max(ref / here, 1) * batch_size))

        # check if batch size is too large
        max_batch_size = np.round(self.batch_size_covers_max_percent_of_dataset * dataset_num_voxels /
                                  np.prod(input_patch_size, dtype=np.int64)).astype(int)
        max_batch_size = max(max_batch_size, self.unet_min_batch_size)
        batch_size = max(1, min(batch_size, max_batch_size))

        do_dummy_2D_data_aug = (max(input_patch_size) / input_patch_size[
            0]) > self.anisotropy_threshold

        plan = {
            'batch_size': batch_size,
            'num_pool_per_axis': network_num_pool_per_axis,
            'patch_size': input_patch_size,
            'median_patient_size_in_voxels': new_median_shape,
            'current_spacing': current_spacing,
            'original_spacing': original_spacing,
            'do_dummy_2D_data_aug': do_dummy_2D_data_aug,
            'pool_op_kernel_sizes': pool_op_kernel_sizes,
            'conv_kernel_sizes': conv_kernel_sizes,
            'num_blocks_encoder': blocks_per_stage_encoder,
            'num_blocks_decoder': blocks_per_stage_decoder
        }
        return plan

    def plan_experiment(self):
        use_nonzero_mask_for_normalization = self.determine_whether_to_use_mask_for_norm()
        print("Are we using the nonzero mask for normalizaion?", use_nonzero_mask_for_normalization)
        spacings = self.dataset_properties['all_spacings']
        sizes = self.dataset_properties['all_sizes']

        all_classes = self.dataset_properties['all_classes']
        modalities = self.dataset_properties['modalities']
        num_modalities = len(list(modalities.keys()))

        target_spacing = self.fixed_spacing[-1]
        original_spacing = self.get_target_spacing()
        original_physical_size = self.get_target_size()
        new_shapes = [np.array(i) for i in sizes]

        max_spacing_axis = np.argmax(target_spacing)
        remaining_axes = [i for i in list(range(3)) if i != max_spacing_axis]
        self.transpose_forward = [max_spacing_axis] + remaining_axes
        self.transpose_backward = [np.argwhere(np.array(self.transpose_forward) == i)[0][0] for i in range(3)]

        # we base our calculations on the median shape of the datasets
        median_shape = np.median(np.vstack(new_shapes), 0)
        print("the median shape of the dataset is ", median_shape)

        max_shape = np.max(np.vstack(new_shapes), 0)
        print("the max shape in the dataset is ", max_shape)
        min_shape = np.min(np.vstack(new_shapes), 0)
        print("the min shape in the dataset is ", min_shape)

        print("we don't want feature maps smaller than ", self.unet_featuremap_min_edge_length, " in the bottleneck")

        # how many stages will the image pyramid have?
        self.plans_per_stage = list()

        target_spacing_transposed = np.array(target_spacing)[self.transpose_forward]
        original_spacing_transposed = np.array(original_spacing)[self.transpose_forward]
        median_shape_transposed = np.array(median_shape)[self.transpose_forward]
        print("the transposed median shape of the dataset is ", median_shape_transposed)

        print("generating configuration for 3d_fullres")
        self.plans_per_stage.append(self.get_properties_for_stage(target_spacing_transposed, original_spacing_transposed,
                                                                  median_shape_transposed,
                                                                  len(self.list_of_cropped_npz_files),
                                                                  num_modalities, len(all_classes) + 1))

        # thanks Zakiyi (https://github.com/MIC-DKFZ/nnUNet/issues/61) for spotting this bug :-)
        # if np.prod(self.plans_per_stage[-1]['median_patient_size_in_voxels'], dtype=np.int64) / \
        #        architecture_input_voxels < HOW_MUCH_OF_A_PATIENT_MUST_THE_NETWORK_SEE_AT_STAGE0:
        architecture_input_voxels_here = np.prod(self.plans_per_stage[-1]['patch_size'], dtype=np.int64)
        if np.prod(median_shape) / architecture_input_voxels_here < \
                self.how_much_of_a_patient_must_the_network_see_at_stage0:
            more = False
        else:
            more = True

        if more:
            print("generating configuration for 3d_lowres")
            # if we are doing more than one stage then we want the lowest stage to have exactly
            # HOW_MUCH_OF_A_PATIENT_MUST_THE_NETWORK_SEE_AT_STAGE0 (this is 4 by default so the number of voxels in the
            # median shape of the lowest stage must be 4 times as much as the network can process at once (128x128x128 by
            # default). Problem is that we are downsampling higher resolution axes before we start downsampling the
            # out-of-plane axis. We could probably/maybe do this analytically but I am lazy, so here
            # we do it the dumb way

            lowres_stage_spacing = self.fixed_spacing[0]
            lowres_stage_spacing_transposed = np.array(lowres_stage_spacing)[self.transpose_forward]

            new = self.get_properties_for_stage_vary(lowres_stage_spacing_transposed, original_spacing_transposed,
                                                     original_physical_size,
                                                     len(self.list_of_cropped_npz_files),
                                                     num_modalities, len(all_classes) + 1)
            num_voxels = np.prod(median_shape, dtype=np.float64)
            if 2 * np.prod(new['median_patient_size_in_voxels'], dtype=np.int64) < np.prod(
                    self.plans_per_stage[0]['median_patient_size_in_voxels'], dtype=np.int64):
                self.plans_per_stage.append(new)

        self.plans_per_stage = self.plans_per_stage[::-1]
        self.plans_per_stage = {i: self.plans_per_stage[i] for i in range(len(self.plans_per_stage))}  # convert to dict

        print(self.plans_per_stage)
        print("transpose forward", self.transpose_forward)
        print("transpose backward", self.transpose_backward)

        normalization_schemes = self.determine_normalization_scheme()
        only_keep_largest_connected_component, min_size_per_class, min_region_size_per_class = None, None, None
        # removed training data based postprocessing. This is deprecated

        # these are independent of the stage
        plans = {'num_stages': len(list(self.plans_per_stage.keys())), 'num_modalities': num_modalities,
                 'modalities': modalities, 'normalization_schemes': normalization_schemes,
                 'dataset_properties': self.dataset_properties, 'list_of_npz_files': self.list_of_cropped_npz_files,
                 'original_spacings': spacings, 'original_sizes': sizes,
                 'preprocessed_data_folder': self.preprocessed_output_folder, 'num_classes': len(all_classes),
                 'all_classes': all_classes, 'base_num_features': self.unet_base_num_features,
                 'use_mask_for_norm': use_nonzero_mask_for_normalization,
                 'keep_only_largest_region': only_keep_largest_connected_component,
                 'min_region_size_per_class': min_region_size_per_class, 'min_size_per_class': min_size_per_class,
                 'transpose_forward': self.transpose_forward, 'transpose_backward': self.transpose_backward,
                 'data_identifier': self.data_identifier, 'plans_per_stage': self.plans_per_stage,
                 'preprocessor_name': self.preprocessor_name,
                 'conv_per_stage': self.conv_per_stage,
                 }

        self.plans = plans
        self.save_my_plans()

