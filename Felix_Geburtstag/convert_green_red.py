import os
from PIL import Image, ImageDraw
import random
import convert_black_white

color = lambda c: ((c >> 16) & 255, (c >> 8) & 255, c & 255)

COLORS_ON = [
    color(0xF9BB82), color(0xEBA170), color(0xFCCD84)
]
COLORS_OFF = [
    color(0x9CA594), color(0xACB4A5), color(0xBBB964),
    color(0xD7DAAA), color(0xE5D57D), color(0xD1D6AF)
]

BACKGROUND= (0, 0, 0)

def overlaps_motive(image, x, y):
    if image.getpixel((x,y)) != BACKGROUND:
        return True
    else:
        return False

def convert(relpath, savepath, thresh = 85):
    img = Image.open(os.getcwd() + "\\" + relpath)
    
    fn = lambda x : 255 if x > thresh else 0
    r = img.convert('L').point(fn, mode='1')
    
    r.save(os.getcwd() + "\\" + savepath + '\\converted.png')
    return savepath + "\\" + 'converted.png'



def convert(relpath, savepath, thresh = 85, num = 0):
    conv = convert_black_white.convert(relpath, "\\trash", thresh)

    img = Image.open(os.getcwd() + conv)
    img = img.convert("RGB")

    pixels = img.load() # create the pixel map

    for i in range(img.size[0]): # for every pixel:
        for j in range(img.size[1]):
            if overlaps_motive(img, i, j):
                fill_colors = COLORS_ON
            else:
                fill_colors = COLORS_OFF
            color = random.choice(fill_colors)
            pixels[i, j] = color
    img.save(os.getcwd() + "\\" + savepath + '\\red_green' + str(num) + '.png')

if __name__ == "__main__":
    convert("Felix_Geburtstag\\Erik_Felixia.jpg", "", 100)
    