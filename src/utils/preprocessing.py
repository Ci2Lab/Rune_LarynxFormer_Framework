from pycocotools.coco import COCO
import torch
from torch.utils.data import random_split 
from torch import tensor 

SUPRAGLOTTIS_ID = 0
TRANCHEA_ID = 1

'''
Convert a COCO mask dictionary to a list of images with annotations.
'''
def preprocess_masks(mask_data):
  # Load COCO masks
  coco_data = COCO(mask_data)
  coco_imgs = coco_data.imgs
  coco_anns = coco_data.anns

  # Connect masks with images
  images_with_annotations = []

  for i in range(len(coco_imgs)):
    images_with_annotations += [coco_imgs[i]]
    images_with_annotations[i]['annotations'] = []

  for i in range(len(coco_anns)):
    if coco_anns[i]['category_id'] != TRANCHEA_ID: continue
    img_id = coco_anns[i]['image_id']
    images_with_annotations[img_id]['annotations'] += [coco_data.annToMask(coco_anns[i])]

  for i in range(len(coco_anns)):
    if coco_anns[i]['category_id'] != SUPRAGLOTTIS_ID: continue
    img_id = coco_anns[i]['image_id']
    images_with_annotations[img_id]['annotations'] += [coco_data.annToMask(coco_anns[i])]

  print(f"Dataset size: {len(images_with_annotations)}")
  return images_with_annotations

'''
Converts a label id to a class index.
'''
def label_mapper(label):
    return {SUPRAGLOTTIS_ID: 2, TRANCHEA_ID: 1}.get(label, 0)

'''
Split a dataset into training, validation and test sets.
'''
def split_dataset(dataset):
  train_set_len = int(len(dataset)*0.7)
  valid_set_len = int(len(dataset)*0.2)
  test_set_len = len(dataset) - train_set_len - valid_set_len

  print(f"Train set length: {train_set_len}")
  print(f"Valid set length: {valid_set_len}")
  print(f"Test set length: {test_set_len}")

  generator = torch.Generator().manual_seed(42)
  train, valid, test = random_split(dataset, [train_set_len, valid_set_len, test_set_len], generator=generator)
  return train, valid, test