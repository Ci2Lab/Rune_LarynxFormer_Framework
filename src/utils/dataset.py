import torch
from utils.plotting import get_img_and_mask
from torchvision.transforms import v2 as T
from torch.utils.data import Dataset
import random

'''
A dataset class for the CLE dataset.
'''
class CleDataset(Dataset):
  def __init__(
      self,
      images_with_annotations,
      data_folder,
      img_size,
      batch_size,
      use_batch_transforms = False,
      mix_batch_transforms = False,
      data_type = torch.float32
  ):
    self.images_with_annotations = images_with_annotations
    self.data_folder = data_folder
    self.img_size = img_size
    self.batch_size = batch_size
    self.use_batch_transforms = use_batch_transforms
    self.mix_batch_transforms = mix_batch_transforms
    self.data_type = data_type

  def __getitem__(self, idx):
    img, mask = get_img_and_mask(self.images_with_annotations, idx, self.data_folder)

    img, mask = self.resize(img, mask)

    if self.use_batch_transforms:
      if self.mix_batch_transforms:
        idx_tfm = (idx + self.batch_size) % len(self.images_with_annotations)
        img_to_tfm, mask_to_tfm = get_img_and_mask(self.images_with_annotations, idx_tfm, self.data_folder)
        img_transformed, mask_transformed = self.transforms(img_to_tfm, mask_to_tfm)
      else:
        img_transformed, mask_transformed = self.transforms(img, mask)
      return [img, img_transformed], [mask, mask_transformed]

    return [img], [mask]

  def __len__(self):
    return len(self.images_with_annotations)
  
  def resize(self, img, mask):
    img = T.Resize(size=self.img_size)(img)
    mask = T.Resize(size=self.img_size)(mask)
    return img, mask
  
  def transforms(self, img, mask):
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)

    img_channels = img.shape[0]
    img_and_mask = torch.cat((img, mask), dim=0)
    transform = T.Compose([
      T.RandomResizedCrop(size=self.img_size, scale=(0.5, 0.6), antialias=True),
      T.RandomHorizontalFlip(p=1),
      T.Resize(size=self.img_size),
      T.ToDtype(self.data_type)
    ])(img_and_mask)

    img_tfm, mask_tfm = transform[:img_channels], transform[img_channels:]

    return img_tfm, mask_tfm
  
def custom_collate(batch):
  images, masks = zip(*batch)
  images = [img for sublist in images for img in sublist]
  masks = [mask for sublist in masks for mask in sublist]
  return [torch.stack(images), torch.stack(masks)]
