# Kaggle Pill Data Preprocessing
### Prerequisites
download pill dataset: https://www.kaggle.com/datasets/perfect9015/pillsdetectiondataset</br>
my local environment has the dataset in `final-project-level3-cv-16/data/kaggle_pill_data`

### Steps to follow
0. Edit noise data</br>
change `pill-man-s-hand-white_114963-1058 (1).xml` to `pill-man-s-hand-white_114963-1058.xml`
1. Create `final-project-level3-cv-16/data/kaggle_pill_data/labels.txt`
```text
tablets
```
2. Create `ann_paths_list.txt` using `1_annotation_file_name_to_txt.py`
```bash
python 1_annotation_file_name_to_txt.py
```
3. Edit .xml files
Need to edit .xml files' path since it is saved in the way the uploader had in it's local desktop </br>
(e.g., `C:\Users\inspiron\Documents\GitHub\TabletDetection\data\images\capsule(1).png`)
```bash
python 2_edit_xml_path.py
```
4. change xml format to json format
```bash
# python 3_xml_to_json.py --ann_paths_list ../data/kaggle_pill_data/ann_paths_list.txt --labels ../data/kaggle_pill_data/labels.txt --output ../data/kaggle_pill_data/train.json
python 3_xml_to_json.py --ann_paths_list ann_paths_list.txt_dir --labels labels.txt_dir --output train.json_dir
```
5. Edit json file </br>
Mac users: cmd+A, cmd+k+f </br>
Win users: ctrl+A, ctrl+k+f </br>
- before
![image](https://user-images.githubusercontent.com/73840274/168699711-8cc96061-bea1-4675-b85e-20d7e00fb22b.png)
- after
![image](https://user-images.githubusercontent.com/73840274/168699729-0ead6849-c2a6-4e9d-a6e5-7c6e49033851.png)

### For More Information
Issue #7: https://github.com/boostcampaitech3/final-project-level3-cv-16/issues/7</br>
Issue #20: https://github.com/boostcampaitech3/final-project-level3-cv-16/issues/20
