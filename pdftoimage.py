
import tempfile
import numpy as np
from pdf2image import convert_from_path




def pdf2Image(src):
    with tempfile.TemporaryDirectory() as tmpdirname:
     print('created temporary directory', tmpdirname)
     images = convert_from_path(src)
     images = list(map(np.array,images))
    #  tmpdirname.cleanup()
     return images
