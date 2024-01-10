import os

from brats_2023_metrics.surface_distance.brats_metrics import get_lesion_wise_results

# Example 1: Get lesion-wise results for a single case
get_lesion_wise_results(
    pred_file_path='',  # Path to the prediction file
    gt_file_path='',  # Path to the ground truth file
    challenge_name='',  # BraTS-GLI, BraTS-SSA, BraTS-MEN, BraTS-PED, BraTS-MET
    output_path='',  # Path to save the results
)


# Example 2: Get lesion-wise results for a folder of predictions and ground truths
# Assumes the following folder structure:
# - data
#   - predictions
#     - BraTS_MET_00001_seg.nii.gz
#     - BraTS_MET_00002_seg.nii.gz
#     - ...
#   - gt
#     - BraTS_MET_00001_seg.nii.gz
#     - BraTS_MET_00002_seg.nii.gz
#     - ...

pred_folder_path = ''  # Path to the folder containing the predictions
gt_folder_path = ''  # Path to the folder containing the ground truths
output_path = ''  # Path to save the results

case_names = [file_name.split('.')[0] for file_name in os.listdir(gt_folder_path)]

print('Processing cases:')

for case_name in case_names:
    print(f'-> {case_name}')

    pred_file_path = os.path.join(pred_folder_path, f'{case_name}.nii.gz')
    gt_file_path = os.path.join(gt_folder_path, f'{case_name}.nii.gz')

    if not os.path.exists(pred_file_path):
        raise ValueError(f'Prediction file {pred_file_path} does not exist.')

    if not os.path.exists(gt_file_path):
        raise ValueError(f'Ground truth file {gt_file_path} does not exist.')

    results_df = get_lesion_wise_results(
        pred_file_path=pred_file_path,
        gt_file_path=gt_file_path,
        challenge_name='BraTS-MET',  # BraTS-GLI, BraTS-SSA, BraTS-MEN, BraTS-PED, BraTS-MET
        output_path=None,
    )

    results_df.insert(0, 'Case_Name', case_name)
    results_df.to_csv(
        os.path.join(output_path, 'brats_metrics_results.csv'),
        header=not os.path.exists(output_path),
        mode='a',
        index=False,
    )
