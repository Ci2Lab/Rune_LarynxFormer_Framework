from IPython.display import display, HTML
import tabulate
import torch

'''
Display the performance metrics in a table format
'''
def display_metrics(lists):
  data = format_list(lists)
  headers = ['Epoch', 'Improved', 'Training loss', 'Validation loss', 'Dice', 'IoU', 'F1', 'Precision', 'Recall', 'Duration']
  table = tabulate.tabulate(tabular_data=data, tablefmt='html', headers=headers, floatfmt=['', '', '.6f', '.6f', '.3f', '.3f', '.3f', '.3f', '', ''])
  display(HTML(table))

def format_float(value, long=False):
  return f"{value:.6f}" if long else f"{value:.3f}"

def format_list(lists):
  t = list(map(list, zip(*lists)))
  for i in range(len(t)):
    for j in range(len(t[i])):
      try:
        x = t[i][j]
        if j == 9:
          duration = t[i][j]
          minutes = int(duration / 60)
          seconds = int(duration % 60)
          t[i][j] = f"{minutes:02}:{seconds:02}"
        elif type(x) is torch.Tensor and x.shape[0] == 2:
          value = t[i][j]
          value_0 = format_float(value[0].item())
          value_1 = format_float(value[1].item())
          t[i][j] = f"{value_0} • {value_1}"
        elif type(x) is float:
          t[i][j] = format_float(x, long=True)
        elif type(x) is bool:
          t[i][j] = "✅" if x else ""
      except Exception as e:
        print('An error occurred during formatting: ', f"Position {i, j}: ", e)
  return t

'''
Convert seconds to a pretty time format. Eg. 01:23:45
'''
def pretty_time(seconds):
  hours = int(seconds / 3600)
  minutes = int((seconds % 3600) / 60)
  seconds = int(seconds % 60)
  return f"{hours:02}:{minutes:02}:{seconds:02}"