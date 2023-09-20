# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 12:59:48 2022

@author: 12593
"""

import numpy as np
import nibabel as nb
import os
from collections import OrderedDict
import pandas as pd
from SurfaceDice import compute_surface_distances, compute_surface_dice_at_tolerance, compute_dice_coefficient
from copy import deepcopy
join = os.path.join


def find_lower_upper_zbound(organ_mask):
    """
    Parameters
    ----------
    seg : TYPE
        DESCRIPTION.

    Returns
    -------
    z_lower: lower bound in z axis: int
    z_upper: upper bound in z axis: int

    """
    organ_mask = np.uint8(organ_mask)
    assert np.max(organ_mask) ==1, print('mask label error!')
    z_index = np.where(organ_mask>0)[2]
    z_lower = np.min(z_index)
    z_upper = np.max(z_index)
    
    return z_lower, z_upper


def get_information(seg_path, save_name):
    seg_metrics = deepcopy(base_seg_metrics)
    for name in filenames:
        seg_metrics['Name'].append(name)
        # load grond truth and segmentation
        gt_nii = nb.load(join(gt_path, name))
        case_spacing = gt_nii.header.get_zooms()
        gt_data = np.uint8(gt_nii.get_fdata())
        if not os.path.exists(join(seg_path, name)):
            return
        seg_data = np.uint8(nb.load(join(seg_path, name)).get_fdata())

        for i, organ in enumerate(label_tolerance.keys(), 1):
            if np.sum(gt_data == i) == 0 and np.sum(seg_data == i) == 0:
                DSC_i = 1
                NSD_i = 1
            elif np.sum(gt_data == i) == 0 and np.sum(seg_data == i) > 0:
                DSC_i = 0
                NSD_i = 0
            else:
                if i == 5 or i == 6 or i == 10:  # for Aorta, IVC, and Esophagus, only evaluate the labelled slices in ground truth
                    z_lower, z_upper = find_lower_upper_zbound(gt_data == i)
                    organ_i_gt, organ_i_seg = gt_data[:, :, z_lower:z_upper] == i, seg_data[:, :, z_lower:z_upper] == i
                else:
                    organ_i_gt, organ_i_seg = gt_data == i, seg_data == i
                surface_distances = compute_surface_distances(organ_i_gt, organ_i_seg, case_spacing)
                DSC_i = compute_dice_coefficient(organ_i_gt, organ_i_seg)
                NSD_i = compute_surface_dice_at_tolerance(surface_distances, label_tolerance[organ])
            seg_metrics['{}_DSC'.format(organ)].append(round(DSC_i, 4))
            seg_metrics['{}_NSD'.format(organ)].append(round(NSD_i, 4))
            print(name, organ, round(DSC_i, 4), 'tol:', label_tolerance[organ], round(NSD_i, 4))

    dataframe = pd.DataFrame(seg_metrics)
    dataframe.to_csv(save_name, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", '--ground_truth', required=True, help="path to ground-truth labels")
    parser.add_argument('-s', "--save_folder", required=True, help="folder for saving information")
    parser.add_argument('-p', "--prediction", required=True, help="path for prediction")

    args = parser.parse_args()
    gt_path = args.ground_truth
    save_path = args.save_folder
    pretiction_path = args.prediction

    # temp_name = 'repvgg_original_patchsize'
    filenames = os.listdir(gt_path)
    filenames = [x for x in filenames if x.endswith('.nii.gz')]
    filenames.sort()
    base_seg_metrics = OrderedDict()
    base_seg_metrics['Name'] = list()
    label_tolerance = OrderedDict({'Liver': 5, 'RK': 3, 'Spleen': 3, 'Pancreas': 5,
                                   'Aorta': 2, 'IVC': 2, 'RAG': 2, 'LAG': 2, 'Gallbladder': 2,
                                   'Esophagus': 3, 'Stomach': 5, 'Duodenum': 7, 'LK': 3, 'Tumor': 2})
    for organ in label_tolerance.keys():
        base_seg_metrics['{}_DSC'.format(organ)] = list()
    for organ in label_tolerance.keys():
        base_seg_metrics['{}_NSD'.format(organ)] = list()

    print(pretiction_path)
    temp_name = os.path.basename(pretiction_path)
    save_name = 'DSC_NSD_%s.csv' % temp_name
    save_name = join(save_path, save_name)
    get_information(pretiction_path, save_name)


if __name__ == "__main__":
    main()





