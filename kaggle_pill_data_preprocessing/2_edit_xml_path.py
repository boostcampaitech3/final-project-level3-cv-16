"""
reference: https://skkim1080.tistory.com/entry/Python%EC%9C%BC%EB%A1%9C-xml-%ED%8C%8C%EC%9D%BC-%EB%82%B4%EC%9A%A9-%EC%88%98%EC%A0%95%ED%95%98%EA%B8%B0
"""

import os
import xml.etree.ElementTree as ET

dir = "../data/kaggle_pill_data"
ann_dir = os.path.join(dir, "annotations")
img_dir = os.path.join(dir, "images")
file_list = os.listdir(ann_dir)
xml_list = []
for file in file_list:
    if ".xml" in file:
        xml_list.append(file)

for xml_file in xml_list:
    path = ann_dir + "/" + xml_file
    targetXML = open(path, "rt", encoding="UTF8")

    tree = ET.parse(targetXML)
    root = tree.getroot()

    target_path, target_filename = root.find("path"), root.find("filename")
    original_path, original_filename = target_path.text, target_filename.text
    modified = os.path.join(img_dir, original_filename)
    target_path.text = modified
    tree.write(path)
