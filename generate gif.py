import imageio
import os

def sort_num(list_of_names):
    list1 = [int(x) for x in list_of_names]
    list1.sort()
    list2 = [str(x) for x in list_of_names]
    return list2

rootdir='D:/Nicolas/Pictures/Pictures for Presentation/validation_images'


images = []
for subdir, dirs, files in os.walk(rootdir):
    for filename in files:
        images.append(imageio.imread(os.path.join(rootdir,filename)))

# filname = sort_num(filname)



kargs = { 'duration': 1 }
save_under= 'D:/Nicolas/Pictures/Pictures for Presentation/validation_images_1.gif'
imageio.mimsave(save_under, images, **kargs)