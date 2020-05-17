import os
import string

src_dir = "/home/csci5980/saluj012/RWF-2000-Original-Data/"
for (root,dirs,files) in os.walk(src_dir, topdown=True):
    for file in files:
        file_name = root +'/' + file
        new_file_name = ''.join(c for c in file_name if c in string.printable)
        os.rename(file_name, new_file_name)

