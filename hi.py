
import shutil
import os

source = 'darknet/custom_dataset/ena24'
destination = 'darknet/custom_dataset/'
files_list = os.listdir(source)
for files in files_list:
    files = 'darknet/custom_dataset/ena24/'+files
    shutil.move(files, destination)