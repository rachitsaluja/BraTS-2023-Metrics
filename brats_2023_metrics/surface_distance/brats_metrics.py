import math
import os.path

import cc3d
import nibabel as nib
import numpy as np
import pandas as pd
import scipy

from brats_2023_metrics import surface_distance


def get_challenge_parameters(challenge_name: str) -> (int, int):
    """Returns the challenge parameters (dilation_factor, lesions_volume_thresh) for the given challenge name

    Args:
        challenge_name (str): Challenge name

    Returns:
        dilation_factor (int): Dilation factor for the challenge
        lesion_volume_thresh (int): Lesion volume threshold for the challenge
    """

    if challenge_name == 'BraTS-GLI':
        dilation_factor = 3
        lesion_volume_thresh = 50
        return dilation_factor, lesion_volume_thresh

    if challenge_name == 'BraTS-SSA':
        dilation_factor = 3
        lesion_volume_thresh = 50
        return dilation_factor, lesion_volume_thresh

    if challenge_name == 'BraTS-MEN':
        dilation_factor = 1
        lesion_volume_thresh = 50
        return dilation_factor, lesion_volume_thresh

    if challenge_name == 'BraTS-PED':
        dilation_factor = 3
        lesion_volume_thresh = 50
        return dilation_factor, lesion_volume_thresh

    if challenge_name == 'BraTS-MET':
        dilation_factor = 1
        lesion_volume_thresh = 2
        return dilation_factor, lesion_volume_thresh

    raise ValueError('Supported challenge names are: BraTS-GLI, BraTS-SSA, BraTS-MEN, BraTS-PED, BraTS-MET')


def get_dice_score(pred_img: np.array, gt_img: np.array) -> float:
    """Computes Dice score for two images

    Args:
        pred_img (np.array): Predicted segmentation as numpy array
        gt_img (np.array): Ground truth segmentation as numpy array

    Returns:
        dice_score (float): Dice score between two images
    """

    pred_array = np.asarray(pred_img).astype(bool)
    gt_array = np.asarray(gt_img).astype(bool)

    if pred_array.shape != gt_array.shape:
        raise ValueError('Shape mismatch: pred_img and gt_img must have the same shape.')

    intersection = np.logical_and(pred_array, gt_array)
    dice_score = 2.0 * (intersection.sum()) / (pred_array.sum() + gt_array.sum())
    return dice_score


def get_tissue_wise_mask(array: np.array, tissue_type: str) -> np.array:
    """Converts array to tissue-wise mask

    Args:
        array (np.array): Predicted segmentation/ground truth segmentation as numpy array
        tissue_type (str): Supported types: WT, ET, TC

    Returns:
        array (np.array): Segmentation array of the tissue type
    """

    if tissue_type == 'WT':
        np.place(array, (array != 1) & (array != 2) & (array != 3), 0)
        np.place(array, (array > 0), 1)
        return array

    if tissue_type == 'TC':
        np.place(array, (array != 1) & (array != 3), 0)
        np.place(array, (array > 0), 1)
        return array

    if tissue_type == 'ET':
        np.place(array, (array != 3), 0)
        np.place(array, (array > 0), 1)
        return array

    raise ValueError('Supported tissue types: WT, TC, ET')


def get_gt_seg_combined_by_dilation(gt_dilated_cc_mat: np.array, gt_label_cc: np.array) -> np.array:
    """Computes the Corrected Connected Components after combing lesions together with respect to their dilation extent

    Args:
        gt_dilated_cc_mat (np.array): Ground Truth Dilated Segmentation after CC Analysis
        gt_label_cc (np.array): Ground Truth Segmentation after CC Analysis

    Returns:
        gt_seg_combined_by_dilation_mat (np.array): Ground Truth Segmentation after CC Analysis and combining lesions
    """

    gt_seg_combined_by_dilation_mat = np.zeros_like(gt_dilated_cc_mat)

    for comp, _ in enumerate(range(np.max(gt_dilated_cc_mat)), start=1):
        gt_d_tmp = np.zeros_like(gt_dilated_cc_mat)
        gt_d_tmp[gt_dilated_cc_mat == comp] = 1
        gt_d_tmp = gt_label_cc * gt_d_tmp
        np.place(gt_d_tmp, gt_d_tmp > 0, comp)
        gt_seg_combined_by_dilation_mat += gt_d_tmp

    return gt_seg_combined_by_dilation_mat


def get_lesion_wise_score(pred_file_path: str, gt_file_path: str, label_name: str, dil_factor: int) -> dict:
    """Computes the Lesion-wise scores for pair of prediction and ground truth segmentations

    Args:
        pred_file_path (str): File path of the prediction image
        gt_file_path (str): File path of the ground truth image
        label_name (str): Supported are: WT, ET, TC
        dil_factor (int): Used to perform dilation

    Returns:
        dict:
            tp (int): Number of TP lesions WRT prediction segmentation
            fn (int): Number of FN lesions WRT prediction segmentation
            fp (int): Number of FP lesions WRT prediction segmentation
            gt_tp (int): Number of Ground Truth TP lesions WRT prediction segmentation
            metric_pairs (list): All the lesion-wise metrics
            full_dice (float): Dice Score of the pair of segmentations
            full_gt_vol (float): Total Ground Truth Segmentation Volume
            full_pred_vol (float): Total Prediction Segmentation Volume
    """

    score_store = {'gt_tp': [], 'tp': [], 'fn': [], 'fp': [], 'metric_pairs': []}

    pred_nii = nib.load(pred_file_path)
    gt_nii = nib.load(gt_file_path)

    pred_array = pred_nii.get_fdata()
    gt_array = gt_nii.get_fdata()

    sx, sy, sz = pred_nii.header.get_zooms()  # Get Spacing to computes volumes, Brats assumes 1x1x1mm3

    pred_array = get_tissue_wise_mask(array=pred_array, tissue_type=label_name)
    gt_array = get_tissue_wise_mask(array=gt_array, tissue_type=label_name)

    if np.all(gt_array == 0) and np.all(pred_array == 0):  # Get Dice score for the full image
        score_store['full_dice'] = 1.0
    else:
        score_store['full_dice'] = get_dice_score(pred_array, gt_array)

    if np.all(gt_array == 0) and np.all(pred_array == 0):  # Get HD95 score for the full image
        score_store['full_hd95'] = 0.0
    else:
        full_sd = surface_distance.compute_surface_distances(gt_array.astype(int), pred_array.astype(int), (sx, sy, sz))
        score_store['full_hd95'] = surface_distance.compute_robust_hausdorff(full_sd, 95)

    results = get_tpr_tnr_ppv_voxel_wise(pred_array=pred_array, gt_array=gt_array)
    (
        score_store['sensitivity_voxel_wise'],
        score_store['specificity_voxel_wise'],
        score_store['precision_voxel_wise'],
    ) = results

    score_store['full_gt_volume'] = np.sum(gt_array) * sx * sy * sz
    score_store['full_pred_volume'] = np.sum(pred_array) * sx * sy * sz

    # Performing Dilation and CC analysis
    dilation_struct = scipy.ndimage.generate_binary_structure(3, 2)

    gt_mat_cc = cc3d.connected_components(gt_array, connectivity=26)
    pred_mat_cc = cc3d.connected_components(pred_array, connectivity=26)

    gt_mat_dilation = scipy.ndimage.binary_dilation(gt_array, structure=dilation_struct, iterations=dil_factor)
    gt_mat_dilation_cc = cc3d.connected_components(gt_mat_dilation, connectivity=26)

    gt_mat_combined_by_dilation = get_gt_seg_combined_by_dilation(
        gt_dilated_cc_mat=gt_mat_dilation_cc,
        gt_label_cc=gt_mat_cc,
    )
    # Performing the Lesion-By-Lesion Comparison
    gt_label_cc = gt_mat_combined_by_dilation
    pred_label_cc = pred_mat_cc

    for gt_comp in range(np.max(gt_label_cc)):
        gt_comp += 1

        # Extracting current lesion
        gt_tmp = np.zeros_like(gt_label_cc)
        gt_tmp[gt_label_cc == gt_comp] = 1

        # Extracting ROI GT lesion component
        gt_tmp_dilation = scipy.ndimage.binary_dilation(gt_tmp, structure=dilation_struct, iterations=dil_factor)

        gt_volume = np.sum(gt_tmp) * sx * sy * sz

        # Extracting Predicted true positive lesions
        pred_tmp = np.copy(pred_label_cc)
        pred_tmp = pred_tmp * gt_tmp_dilation
        intersecting_cc = np.unique(pred_tmp)
        intersecting_cc = intersecting_cc[intersecting_cc != 0]
        for cc in intersecting_cc:
            score_store['tp'].append(cc)

        # Isolating Predited Lesions to calulcate Metrics
        pred_tmp = np.copy(pred_label_cc)
        pred_tmp[np.isin(pred_tmp, intersecting_cc, invert=True)] = 0
        pred_tmp[np.isin(pred_tmp, intersecting_cc)] = 1

        # Calculating Lesion-wise Dice and HD95
        dice_score = get_dice_score(pred_tmp, gt_tmp)
        surface_distances = surface_distance.compute_surface_distances(gt_tmp, pred_tmp, (sx, sy, sz))
        hd = surface_distance.compute_robust_hausdorff(surface_distances, 95)

        score_store['metric_pairs'].append((intersecting_cc, gt_comp, gt_volume, dice_score, hd))

        # Extracting Number of TP/FP/FN and other data
        if len(intersecting_cc) > 0:
            score_store['gt_tp'].append(gt_comp)
        else:
            score_store['fn'].append(gt_comp)

    score_store['fp'] = np.unique(pred_label_cc[np.isin(pred_label_cc, score_store['tp'] + [0], invert=True)])
    return score_store


def get_tpr_tnr_ppv_voxel_wise(pred_array: np.array, gt_array: np.array) -> (float, float, float):
    """This function is extracted from GaNDLF from mlcommons
    https://github.com/mlcommons/GaNDLF/blob/master/GANDLF/metrics/segmentation.py#L196

    Args:
        pred_array (np.array): Predicted segmentation as numpy array
        gt_array (np.array): Ground truth segmentation as numpy array

    Returns:
        sensitivity (float): Sensitivity score
        specificity (float): Specificity score
        precision (float): Precision score
    """
    pred_count = int(np.sum(pred_array))
    gt_count = int(np.sum(gt_array))

    overlap = np.where((pred_array == gt_array), 1, 0)  # Where they agree are both equal to that value

    tp = int(overlap[pred_array == 1].sum())
    fp = pred_count - tp
    fn = gt_count - tp
    tn = np.count_nonzero((pred_array != 1) & (gt_array != 1))

    sensitivity = tp / (tp + fn + np.finfo(float).eps)
    specificity = tn / (tn + fp + np.finfo(float).eps)
    precision = tp / (tp + fp + np.finfo(float).eps)

    if (pred_count == 0) and (gt_count == 0):  # Make Changes if both input and reference are 0 for the tissue type
        sensitivity = 1.0

    return sensitivity, specificity, precision


def get_lesion_wise_results(
    pred_file_path: str, gt_file_path: str, challenge_name: str, output_path: str = None
) -> pd.DataFrame:
    """Computes the Lesion-wise scores for pair of prediction and ground truth segmentations

    Args:
        pred_file_path (str): File path of the prediction image
        gt_file_path (str): File path of the ground truth image
        challenge_name (str): Name of the challenge
        output_path (str): File path to save the results

    Returns:
        Saves the performance metrics as CSVs
        results_df: pd.DataFrame; lesion-wise results with other metrics
    """

    dilation_factor, lesion_volume_thresh = get_challenge_parameters(challenge_name=challenge_name)
    final_metrics_dict = {}
    label_names = ['WT', 'TC', 'ET']

    for label_name in label_names:
        score_store = get_lesion_wise_score(
            pred_file_path=pred_file_path,
            gt_file_path=gt_file_path,
            label_name=label_name,
            dil_factor=dilation_factor,
        )
        metric_df = (
            pd.DataFrame(
                score_store['metric_pairs'],
                columns=[
                    'predicted_lesion_numbers',
                    'gt_lesion_numbers',
                    'gt_lesion_vol',
                    'dice_lesion_wise',
                    'hd95_lesion_wise',
                ],
            )
            .sort_values(by=['gt_lesion_numbers'], ascending=True)
            .reset_index(drop=True)
        )

        metric_df['_len'] = metric_df['predicted_lesion_numbers'].map(len)

        # Removing lesions with below threshold volume
        fn_sub = (metric_df[(metric_df['_len'] == 0) & (metric_df['gt_lesion_vol'] <= lesion_volume_thresh)]).shape[0]

        gt_tp_sub = (metric_df[(metric_df['_len'] != 0) & (metric_df['gt_lesion_vol'] <= lesion_volume_thresh)]).shape[
            0
        ]

        metric_df['Label'] = [label_name] * len(metric_df)
        metric_df = metric_df.replace(np.inf, 374)

        metric_df_thresh = metric_df[metric_df['gt_lesion_vol'] > lesion_volume_thresh]

        try:
            lesion_wise_dice = np.sum(metric_df_thresh['dice_lesion_wise']) / (
                len(metric_df_thresh) + len(score_store['fp'])
            )
        except:
            lesion_wise_dice = np.nan

        try:
            lesion_wise_hd95 = (np.sum(metric_df_thresh['hd95_lesion_wise']) + len(score_store['fp']) * 374) / (
                len(metric_df_thresh) + len(score_store['fp'])
            )
        except:
            lesion_wise_hd95 = np.nan

        if math.isnan(lesion_wise_dice):
            lesion_wise_dice = 1

        if math.isnan(lesion_wise_hd95):
            lesion_wise_hd95 = 0

        tp_lesion_wise = len(score_store['tp']) - gt_tp_sub
        fp_lesion_wise = len(score_store['fp'])
        fn_lesion_wise = len(score_store['fn']) - fn_sub

        sensitivity_lesion_wise = tp_lesion_wise / (tp_lesion_wise + fn_lesion_wise + np.finfo(float).eps)
        precision_lesion_wise = tp_lesion_wise / (tp_lesion_wise + fp_lesion_wise + np.finfo(float).eps)

        metrics_dict = {
            'Num_TP': len(score_store['gt_tp']) - gt_tp_sub,  # GT_TP
            'Num_FP': len(score_store['fp']),
            'Num_FN': len(score_store['fn']) - fn_sub,
            'Sensitivity_Lesion_Wise': sensitivity_lesion_wise,
            'Precision_Lesion_Wise': precision_lesion_wise,
            'Sensitivity_Voxel_Wise': score_store['sensitivity_voxel_wise'],
            'Specificity_Voxel_Wise': score_store['specificity_voxel_wise'],
            'Precision_Voxel_Wise': score_store['precision_voxel_wise'],
            'Legacy_Dice': score_store['full_dice'],
            'Legacy_HD95': score_store['full_hd95'],
            'GT_Complete_Volume': score_store['full_gt_volume'],
            'Lesion_Wise_Score_Dice': lesion_wise_dice,
            'Lesion_Wise_Score_HD95': lesion_wise_hd95,
        }

        final_metrics_dict[label_name] = metrics_dict

    results_df = pd.DataFrame(final_metrics_dict).T
    results_df['Labels'] = results_df.index
    results_df = results_df.reset_index(drop=True)
    results_df.insert(0, 'Labels', results_df.pop('Labels'))
    results_df.replace(np.inf, 374, inplace=True)  # Replace Inf with 374 for HD95

    if output_path:
        results_df.to_csv(os.path.join(output_path, 'brats_metrics_results.csv'), index=False)

    return results_df
