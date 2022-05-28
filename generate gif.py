import imageio
import os

rootdir='/home/rns4fe/Pictures/Pictures_for_Presentation/validation_image_60_x123-swin_different_perspectives_2_cropped/'


images = []
for subdir, dirs, files in os.walk(rootdir):
    for filename in files:
        images.append(imageio.imread(os.path.join(rootdir,filename)))
imageio.mimsave('/home/rns4fe/Pictures/Pictures_for_Presentation/validation_image_60_x123-swin_different_perspectives_2.gif', images)