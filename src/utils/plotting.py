from pathlib import Path
import cv2
import torch
from torch import tensor
import matplotlib.pyplot as plt

DATA_TYPE = torch.float32

'''
Get an image as a tensor from a filename and folder.
'''
def get_img(filename, folder):
    filename_path = Path(filename)
    path_str = str(folder/filename_path.name)
    img = cv2.imread(path_str)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t = tensor(img, dtype=DATA_TYPE)
    return t

'''
Plot an image on the left and image with a segmentation mask to the right.
'''
def plot_image_with_mask(img, mask):
    _, axs = plt.subplots(1, 2, figsize=(8, 2))
    img_permute = img.permute(1, 2, 0)
    img = img_permute / 255.0
    cmap_1 = plt.get_cmap('viridis', 6)
    cmap_2 = plt.get_cmap('viridis', 6)
    mask_1 = mask[1]
    mask_2 = mask[2]

    axs[0].imshow(img)
    axs[0].set_title('Image')
    axs[0].axis('off')

    axs[1].imshow(img, cmap='gray', alpha=1)
    axs[1].imshow(mask_1, cmap=cmap_1, alpha=0.4)
    axs[1].imshow(mask_2, cmap=cmap_2, alpha=0.4)
    axs[1].set_title('Image with Mask')
    axs[1].axis('off')

    plt.show()

'''
Find an image and mask from a dataset and folder.
'''
def get_img_and_mask(dataset, idx, folder):
    image_with_annotations = dataset[idx]
    file_name = image_with_annotations["file_name"]
    img = get_img(file_name, folder)
    img = img.permute(2, 0, 1) # Shifts shape from (h, w, c) to (c, h, w)
    
    mask_0 = tensor(image_with_annotations["annotations"][0], dtype=DATA_TYPE)
    mask_1 = tensor(image_with_annotations["annotations"][1], dtype=DATA_TYPE)
    not_background = torch.clamp(mask_0 + mask_1, 0, 1)
    background = 1 - not_background
    mask = torch.stack((background, mask_0, mask_1), dim=0)

    return img, mask

'''
Plot a list of images
'''
def plot_images(images):
    # Calculate the number of rows and columns based on the desired layout
    num_rows = len(images) // 6 + int(len(images) % 6 > 0)
    num_cols = min(len(images), 6)

    _, axes = plt.subplots(num_rows, num_cols, figsize=(10, num_rows))

    for i, img in enumerate(images):
        img_permute = img.permute(1, 2, 0)
        img = img_permute / 255.0
        
        # If there is only one row, axes is 1D, otherwise 2D
        if num_rows == 1:
            axes[i].imshow(img)
            axes[i].axis('off')
        else:
            axes[i // num_cols, i % num_cols].imshow(img)
            axes[i // num_cols, i % num_cols].axis('off')

    # Adjust layout to prevent overlapping
    plt.tight_layout(pad = 0)
    plt.show()

'''
Print the images in a dataloader
'''
def print_dataset(dataloader):
  data_iterator = iter(dataloader)
  images, _ = next(data_iterator)

  image_tensors = []

  for _ in range(len(dataloader)-1):
    for j in range(len(images)):
      img = images[j]
      image_tensors.append(img)
    images, _ = next(data_iterator)
  
  plot_images(image_tensors)

'''
Find an image with annotations from a dataset and folder.
'''
def get_img_and_annotations(dataset, idx, folder):
    image_with_annotations = dataset[idx]
    file_name = image_with_annotations["file_name"]
    img = get_img(file_name, folder)
    img = img.permute(2, 0, 1) # Shifts shape from (h, w, c) to (c, h, w)
    anns = image_with_annotations["annotations"]
    return img, anns