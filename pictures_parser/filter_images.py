##This script resizes all the images to 128x128x3 format##
from PIL import Image
from os import listdir
from os.path import isfile, join

all_files, all_paths, exist_filter = [], [], []
mypath = './london2018_pics'
filterpath = './london2018_pics'
all_files.extend([f for f in listdir(mypath) if isfile(join(mypath, f))])
all_paths.extend(['./london2018_pics/'+f for f in listdir(mypath) if isfile(join(mypath, f))])
#exist_filter.extend([f for f in listdir(filterpath) if isfile(join(filterpath, f))])

## adjust width and height to your needs
width = 128
height = 128

for path,file in zip(all_paths,all_files):
#    if path != './all_labels/.DS_Store' and file not in exist_filter:
    im = Image.open(path)
    new_im = im.resize((width, height), Image.ANTIALIAS)
    new_im.save("./filtered_london_pics/"+file)