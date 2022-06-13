"""
reference:
https://seong6496.tistory.com/100
https://seong6496.tistory.com/108
"""

import os

dir = "../data/kaggle_pill_data"
ann_dir = os.path.join(dir, "annotations")
file_lists = os.listdir(ann_dir)

new_file_list = []
for file_list in file_lists:
    new_file_list.append(os.path.join(ann_dir, file_list))

with open(os.path.join(dir, "ann_paths_list.txt"), "w", encoding="UTF-8") as f:
    for list in new_file_list:
        f.write(list + "\n")
