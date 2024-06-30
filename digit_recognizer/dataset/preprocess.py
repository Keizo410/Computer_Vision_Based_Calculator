# Delete all of the images that contain non number and _ in the name
from curses.ascii import isalpha
import os
from tokenize import String

folder_path = "./dataset/validation"

file_c = 0

for class_list in os.listdir(folder_path):

    if file_c != 0:
        print(f"file num for {class_list}", file_c)

    i = 0
    file_c=0
    new_path = os.path.join(folder_path, class_list)

    for image_name in os.listdir(new_path):
        i += 1
    
    file_c=i


