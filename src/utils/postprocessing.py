import numpy as np
import cv2
import torch

'''
Smooth the edges of the masks
'''
def smooth_mask_edges(masks):
  masks_np = masks.detach().numpy()
  smoothed_mask = np.zeros_like(masks_np)

  for img in range(masks_np.shape[0]):
     # Trachea
     smoothed_mask[img, 1] = cv2.morphologyEx(masks_np[img, 1], cv2.MORPH_ERODE, np.ones((2, 2),np.uint8), iterations=2)

     # Supraglottis
     smoothed_mask[img, 2] = cv2.morphologyEx(masks_np[img, 2], cv2.MORPH_DILATE, np.ones((10, 10),np.uint8), iterations=5)
     smoothed_mask[img, 2] = cv2.morphologyEx(masks_np[img, 2], cv2.MORPH_CLOSE, np.ones((3, 3),np.uint8), iterations=5)


  return torch.tensor(smoothed_mask, dtype=torch.float32)

'''
Remove outliers from the masks
'''
def remove_mask_outliers(masks):
  masks_np = masks.detach().numpy()
  masks_rounded = np.round(masks_np).astype(np.uint8)

  for img in range(masks_rounded.shape[0]):
    for class_id in range(masks_rounded.shape[1]):
      (numLabels, labels, stats, _) = cv2.connectedComponentsWithStats(masks_rounded[img, class_id], connectivity=4)
      labels_with_largest_area = np.flip(np.argsort(stats[:, cv2.CC_STAT_AREA]))
      if numLabels > 2:
        for label in labels_with_largest_area[2:]:
          masks_rounded[img, 2][labels == label] = 0

  return torch.tensor(masks_rounded, dtype=torch.float32)

'''
Postprocessing of masks
'''
def postprocess_masks(masks):
  masks = smooth_mask_edges(masks)
  masks = remove_mask_outliers(masks)
  return masks