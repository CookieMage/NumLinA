import os
from PIL import Image
import convert_black_white
import Ishahara_test_generator
import make_transparent

NUM = 10000
LAYERS = 20
BACKGROUND = (255, 255, 255)

cwd = os.getcwd()

ishahara = []
trans = []
thresh = []

for i in range(LAYERS):
    ishahara += ["trash\\ishahara" + str(i) + ".png"]
    trans += ["trash\\trans" + str(i) + ".png"]
    thresh += [i*(100/LAYERS)]

for i,e in enumerate(ishahara):
    converted = convert_black_white.convert("Felix_Geburtstag\\Erik_Felixia.jpg", "\\trash", thresh[i])
    _ = Ishahara_test_generator.generate(converted, (i+1)*NUM//LAYERS, e, (0.5*LAYERS/(i+LAYERS*0.5)))
    _ = make_transparent.trans(e, trans[i])
    os.remove(cwd + "\\" + e)

foreground = Image.open(os.getcwd() + "\\" + trans[i])
background = Image.new('RGB', foreground.size, BACKGROUND)

#background = Image.open(os.getcwd() + "\\Felix_Geburtstag\\Erik_Felixia.jpg")

for i in range(0, LAYERS):
    foreground = Image.open(os.getcwd() + "\\" + trans[i])

    background.paste(foreground, (0, 0), foreground)
background.save('done.png')
