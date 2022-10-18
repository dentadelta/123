import pandas as pd
from IPython.core.display import HTML

def callback(x,filename=None):
    def path_to_image_html(path):
        return '<img src="' + path + '" width="100" >'
    x =x.to_html(escape=False,formatters=dict(photo=path_to_image_html))
    if fileName is not None:
      with open(fileName,'w') as file:
        file.write(x)
        file.close()
    return HTML(x)
  
  # do something with df:
  # df = pd.read_csv('...')
  #callback(df)
