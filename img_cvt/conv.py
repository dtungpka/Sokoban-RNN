#Read all png in current dir, convert them to bmp

import os
import sys

from PIL import Image

def convert_to_bmp(path):
    for filename in os.listdir(path):
        if filename.endswith(".png"):
            im = Image.open(os.path.join(path, filename))
            im.save(os.path.join(path,'out', filename[:-4] + ".bmp"))

if __name__ == "__main__":
    path = sys.argv[1]
    convert_to_bmp(path)