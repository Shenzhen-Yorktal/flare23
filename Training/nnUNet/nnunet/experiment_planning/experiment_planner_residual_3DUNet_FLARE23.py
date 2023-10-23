
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


class ExperimentPlanner3DFabiansResUNet_v21_Tumor(ExperimentPlanner3DFabiansResUNet_v21):
    def __init__(self, folder_with_cropped_data, preprocessed_output_folder, only_lowres=False):
        super(ExperimentPlanner3DFabiansResUNet_v21_Tumor, self).__init__(folder_with_cropped_data,
                                                                    preprocessed_output_folder, only_lowres)
        self.data_identifier = "ExperimentPlanner3DFabiansResUNet_v21_Tumor"  # "nnUNetData_FabiansResUNet_v2.1"
        self.plans_fname = join(self.preprocessed_output_folder,
                                "ExperimentPlanner3DFabiansResUNet_v21_Tumor_plans_3D.pkl")
        self.preprocessor_name = 'GenericPreprocessor'

    def get_target_spacing(self):
        target = [4.0, 1.2, 1.2]
        return target


class ExperimentPlanner3DFabiansResUNet_v21_Organs(ExperimentPlanner3DFabiansResUNet_v21):
    def __init__(self, folder_with_cropped_data, preprocessed_output_folder, only_lowres=False):
        super(ExperimentPlanner3DFabiansResUNet_v21_Organs, self).__init__(folder_with_cropped_data,
                                                                    preprocessed_output_folder, only_lowres)
        self.data_identifier = "ExperimentPlanner3DFabiansResUNet_v21_Organs"  # "nnUNetData_FabiansResUNet_v2.1"
        self.plans_fname = join(self.preprocessed_output_folder,
                                "ExperimentPlanner3DFabiansResUNet_v21_Organs_plans_3D.pkl")
        self.preprocessor_name = 'GenericPreprocessor'

    def get_target_spacing(self):
        target = [2.5, 0.82, 0.82]
        return target
