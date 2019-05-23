#This script cleans the broken images in the original dataset
import shutil
from os import listdir
from os.path import isfile, join
import pandas as pd

all_files = []
labels = list(pd.read_csv('./denver_filtered.csv').subDirectory_filePath)
proper_files = [f.split('/')[1] for f in labels]


mypath = './all_labels'
all_files.extend([f for f in listdir(mypath) if isfile(join(mypath, f))])

for i,file in enumerate(all_files):
    if file not in proper_files:
       shutil.move('./all_labels/'+file, "./other_labels")