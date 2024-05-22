import datetime
from IPython import get_ipython

'''
Return the filename of the current notebook.
'''
def get_nb_filename():
  ip = get_ipython()
  path = None
  if '__vsc_ipynb_file__' in ip.user_ns:
      path = ip.user_ns['__vsc_ipynb_file__']
  else: return None
  return path.split('\\')[-1]

'''
Generates a model name based on the filename of the notebook and the current date.
'''
def generate_model_name(filename):
  split = filename.split('_')
  model_name = split[0]
  structure = split[1]
  datestr = datetime.datetime.now().strftime("%Y%m%d")
  return f"{datestr[2:]}_{structure}_{model_name}.pt"
