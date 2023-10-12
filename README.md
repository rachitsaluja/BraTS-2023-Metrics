# BraTS 2023 Lesion-Wise Performance Metrics for Segmentation Challenges

## Outline

Thank you all for participating in this year's BraTS Challenge. 

This year we are introducing 2 new performance metrics called lesion-wise dice score and lesion-wise Hausdorff distance-95 (HD95). This is mainly developed to understand the performance of a model at a lesion level and not at an image level. By evaluating models lesion-by-lesion we can understand how well models catch and segment abnormalities, and doesn't bias the results in favor of models that capture only large lesions. 

Below is an outline of how we perform this analysis - 

1.  First, we isolate the Lesion Tissues into WT; TC and ET.
2.  We perform a dilation on the Ground Truth (GT) labels (for WT; TC and ET) to understand the extent of the lesion. This is mainly done so that when we do connected component analysis; we don't classify small lesions near an "actual" lesion as a new one. An important thing to note is that the GT labels doesn't change in the process.
3.  We perform connected component analysis on the Prediction label and compare it component by component to the GT label.
4.  We calculate dice scores and HD95 scores for each lesion (or component) individually and we penalize all the False Positives and the False Negatives with a 0 score. Then, we take the mean for the particular CaseID.
5.  Each challenge leader has set a volumetric threshold, below which participants' models won't be evaluated for those "small/false" lesions.
 
