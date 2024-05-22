import torch
from torch import tensor
from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryF1Score

'''
A class to calculate the Dice score for a matrix of predictions and targets
'''
class Dice():
    "Dice coefficient metric for binary target in segmentation"
    def __init__(self, axis=1):
      self.axis = axis
      self.inter, self.union = 0, 0
    def reset(self):
      self.inter, self.union = 0, 0
    def accumulate(self, pred, y):
      prediction, target = pred.argmax(dim=1), y[:,self.axis,:]
      prediction = torch.where(prediction == self.axis, 1, 0)
      self.inter += torch.sum(prediction * target).item()
      self.union += torch.sum(prediction).item() + torch.sum(target).item()
    def get_value_and_reset(self):
      result = self.value
      self.reset()
      return result

    @property
    def value(self): return 2. * self.inter/self.union if self.union > 0 else None

'''
A class to calculate the Intersection over Union for a matrix of predictions and targets
'''
class IoU():
    "IoU coefficient metric for binary target in segmentation"
    def __init__(self, axis=1):
      self.axis = axis
      self.inter, self.union = 0, 0
    def reset(self):
      self.inter, self.union = 0, 0
    def accumulate(self, pred, y):
      prediction, target = pred.argmax(dim=1), y[:,self.axis,:]
      prediction = torch.where(prediction == self.axis, 1, 0)
      self.inter += torch.sum(prediction * target).item()
      self.union += torch.sum((prediction + target) > 0).item()
    def get_value_and_reset(self):
      result = self.value
      self.reset()
      return result

    @property
    def value(self): return self.inter/self.union if self.union > 0 else None

'''
Calculate the precision, recall and F1 score for a matrix of predictions and targets
'''
def calculate_metrics(pred, y, dim, device):
    pred = pred.argmax(dim=1)
    pred = torch.where(pred == dim, 1, 0)
    y = y[:,dim,:].int()
    precision_fn = BinaryPrecision().to(device)
    recall_fn = BinaryRecall().to(device)
    f1_fn = BinaryF1Score().to(device)
    return precision_fn(pred, y), recall_fn(pred, y), f1_fn(pred, y)

'''
Generate a list of empty tensors
'''
def get_empty_tensors(length):
  return (torch.zeros(2, dtype=torch.float32) for _ in range(length))

'''
A class for calculating metrics for a prediction and target. Includes Dice, IoU, Precision, Recall and F1 score.
'''
class Metrics():
  def __init__(self, device):
    self.device = device
    self.precision, self.recall, self.f1 = get_empty_tensors(3)
    self.dice_1 = Dice(axis=1)
    self.dice_2 = Dice(axis=2)
    self.iou_1 = IoU(axis=1)
    self.iou_2 = IoU(axis=2)
  def reset(self):
    self.precision, self.recall, self.f1 = get_empty_tensors(3)
  def accumulate(self, pred, y):
    self.dice_1.accumulate(pred, y)
    self.dice_2.accumulate(pred, y)
    self.iou_1.accumulate(pred, y)
    self.iou_2.accumulate(pred, y)
    precision_1, recall_1, f1_1 = calculate_metrics(pred, y, dim=1, device=self.device)
    precision_2, recall_2, f1_2 = calculate_metrics(pred, y, dim=2, device=self.device)
    self.precision += tensor((precision_1, precision_2))
    self.recall += tensor((recall_1, recall_2))
    self.f1 += tensor((f1_1, f1_2))
  def get_value_and_reset(self, n_batches):
    p_final, r_final, f1_final = self.precision, self.recall, self.f1
    dice_tranchea = self.dice_1.get_value_and_reset()
    dice_supraglottis = self.dice_2.get_value_and_reset()
    iou_tranchea = self.iou_1.get_value_and_reset()
    iou_supraglottis = self.iou_2.get_value_and_reset()
    dice_score = tensor((dice_tranchea, dice_supraglottis))
    iou_score = tensor((iou_tranchea, iou_supraglottis))
    self.reset()
    return dice_score, iou_score, p_final / n_batches, r_final / n_batches, f1_final / n_batches