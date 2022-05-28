import sys
import os
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import image
from PIL import Image
import numpy as np



rootdir = '/home/rns4fe/Pictures/Pictures_for_Presentation/validation_image_60_x123-swin_different_perspectives_2'

def plot_image(image):
    image = np.array(image)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(25, 15));
    fig.tight_layout()
    ax.imshow(image)
    plt.show()

for subdir, dirs, files in os.walk(rootdir):
    for filename in files:

        # if "Screenshot" not in filename:
        #     continue
        ending = filename.split(".")[1]
        if ending not in ["png"]:
            continue

        try:
            image = Image.open(os.path.join(subdir, filename))
        except IOError as e:
            print("Problem Opening", subdir, ":", e)
            continue

        #                 left, upper, right, lower
        image = image.crop((100, 100, 1910, 1070))

        # plot_image(image)

        # name, extension = os.path.splitext(filename)
        print(subdir, filename + '_cropped.png')
        # plt.savefig(os.path.join(subdir, filename + '_cropped.png'), dpi=300)
        image = image.save(os.path.join(
            '/home/rns4fe/Pictures/Pictures_for_Presentation/validation_image_60_x123-swin_different_perspectives_2_cropped'
            , filename))