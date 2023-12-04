import os
from PIL import Image

def convert(relpath, savepath, thresh = 85):
    img = Image.open(os.getcwd() + "\\" + relpath)
    
    fn = lambda x : 255 if x > thresh else 0
    r = img.convert('L').point(fn, mode='1')
    
    r.save(os.getcwd() + "\\" + savepath + '\\converted.png')
    return savepath + "\\" + 'converted.png'

if __name__ == "__main__":
    convert("Felix_Geburtstag\\Erik_Felixia.jpg", "\\trash", 100)
    