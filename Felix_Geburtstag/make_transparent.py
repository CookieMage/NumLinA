import os
from PIL import Image

def trans(relpath, name):
    img = Image.open(os.getcwd() + "\\" + relpath)
    rgba = img.convert("RGBA")
    datas = rgba.getdata()

    new_data = []
    for item in datas:
        if item[0] == 255 and item[1] == 255 and item[2] == 255:  # finding white colour by its RGB value
            # storing a transparent value when we find a white colour
            new_data.append((255, 255, 255, 0))
        else:
            new_data.append(item)  # other colours remain unchanged

    rgba.putdata(new_data)
    rgba.save(name)
    return name
