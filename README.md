# 너의 알약이 보여 - by. Team Medic(CV-16)

---

## 📚 Project Overview

- Project Period: 2022.04.07 ~ 2022.06.10
- Project Presentation Video: [Link to YouTube](https://www.youtube.com/watch?v=lCG4DU4Wljc)

- Project Presentation File: [CV_16조_알약분류_최종프로젝트 발표자료.pdf](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/66c8546c-34bf-433c-a44a-f598594ef9ad/CV_16조_알약분류_최종프로젝트_발표자료.pdf)

- Project Wrap-up Report: [최종 프로젝트_CV_16_Wrap UP Report.pdf](https://github.com/boostcampaitech3/final-project-level3-cv-16/files/8896773/_CV_16_Wrap.UP.Report.pdf)


## 👀 너의 알약이 보여 💊

- Metric Learning을 활용한 Reverse Pill Image Search
- streamlit 실행 예시

![프로젝트 시연.gif](https://github.com/boostcampaitech3/final-project-level3-cv-16/blob/develop/src/streamlit%20%E1%84%91%E1%85%B3%E1%84%85%E1%85%A9%E1%84%8C%E1%85%A6%E1%86%A8%E1%84%90%E1%85%B3%20%E1%84%89%E1%85%B5%E1%84%8B%E1%85%A7%E1%86%AB.gif)

### 😎 Members

| [권순호](https://github.com/tnsgh9603) | [서다빈](https://github.com/sodabeans) | [서예현](https://github.com/justbeaver97) | [이상윤](https://github.com/SSANGYOON) | [전경민](https://github.com/seoulsky-field) |
| --- | --- | --- | --- | --- |
| ![권순호](https://github.com/boostcampaitech3/final-project-level3-cv-16/blob/develop/src/%E1%84%80%E1%85%AF%E1%86%AB%E1%84%89%E1%85%AE%E1%86%AB%E1%84%92%E1%85%A9.png) | ![서다빈](https://github.com/boostcampaitech3/final-project-level3-cv-16/blob/develop/src/%E1%84%89%E1%85%A5%E1%84%83%E1%85%A1%E1%84%87%E1%85%B5%E1%86%AB.png) | ![서예현](https://github.com/boostcampaitech3/final-project-level3-cv-16/blob/develop/src/%E1%84%89%E1%85%A5%E1%84%8B%E1%85%A8%E1%84%92%E1%85%A7%E1%86%AB.jpg) | ![이상윤](https://github.com/boostcampaitech3/final-project-level3-cv-16/blob/develop/src/%E1%84%8B%E1%85%B5%E1%84%89%E1%85%A1%E1%86%BC%E1%84%8B%E1%85%B2%E1%86%AB.png) | ![전경민](https://github.com/boostcampaitech3/final-project-level3-cv-16/blob/develop/src/%E1%84%8C%E1%85%A5%E1%86%AB%E1%84%80%E1%85%A7%E1%86%BC%E1%84%86%E1%85%B5%E1%86%AB.png) |

### 🤗 Contribution

- 권순호: FastAPI, BentoML, streamlit, GCP, OCR, Text Recognition
- 서다빈: FastAPI, streamlit, OCR, Text Recognition
- 서예현: Data EDA, Data Pre-processing, Image Classification, Custom Dataset Production
- 이상윤: Metric learning, Segmentation, Database, Docker
- 전경민: Data EDA, Data Pre-processing, Data Annotation, OCR, Text Recognition

## **❓ About This Project**

### Purpose

- 사용자의 알약 이미지로부터 알약을 식별하는 인공지능 서비스

### Objective

- 사용자의 **알약 이미지**로부터 **성상, 제형, 색상**을 식별 후 조건에 맞는 알약을 검색하여 알약의 **종류**를 식별한다.

### Target Audience

- 지리적, 물리적 한계로 약국이나 병원을 방문하기 어려운 사람
- 알약은 있지만 알약을 구분할 수 없는 사람

### Background Information

- 종종 일어나는 처방 실수, 및 착각으로 인한 약물사고를 예방하고자 하였다.
- 실제 보건 계열 종사자에 따르면 노년 층의 경우 어떤 알약인지 병원에 방문하여 알약을 찾는 경우가 존재한다고 하며, 한국의 통계를 보았을 때도 약물 오복용에 의한 사고는 줄지 않고 계속 유지되고 있는 추세이다.

## 🗺 Service Architecture

![Service Architecture](https://github.com/boostcampaitech3/final-project-level3-cv-16/blob/develop/src/Service%20Architecture.png)

## 💾 Datasets

- 의약품 안전나라 데이터 ([Link](https://nedrug.mfds.go.kr/pbp/CCBGA01/getItem?totalPages=4&limit=10&page=2&&openDataInfoSeq=11))
- ePillID Benchmark ([Link](https://github.com/usuyama/ePillID-benchmark))
- 기타 이미지 데이터 ([Link](https://unsplash.com/s/photos/pill))
    - Classification 및 Object Detection을 위해 직접 촬영 및 수집한, 라이센스가 없는 이미지들

## 💻 **Development Environment**

- GPU: Tesla V100
- OS: Ubuntu 18.04.5LTS
- CPU: Intel Xeon
- Python : 3.8.5 / 3.9.13

## 📁 Project Structure (Main branch)

```markdown
final-project-level3-cv-16
├─ api_folder
│   ├─ .streamlit
|   |   └─ config.toml
│   ├─ backend
|   |   ├─ epillid_benchmark(cloned from Link)
|   |   ├─ Dockerfile
|   |   ├─ Backend.py
|   |   └─ requirements.txt
│   ├─ frontend
|   |   ├─ Dockerfile
|   |   ├─ frontend.py
|   |   └─ requirements.txt
│   └─ Docker
|       └─ docker-compose.yml  
└─ image_classification
    ├─ data_preprocessing
    |   ├─ download_pill_data.py 
    |   └─ normalize_pill_data.py
    ├─ image_concatenation
    |   └─ concatenation_images.py
    ├─ kaggle_pill_data_preprocessing
    |   ├─ 1_annotation_file_name_to_txt.py
    |   ├─ 2_edit_xml_path.py
    |   └─ 3_xml_to_json.py
    ├─ pill_excel_data
    |   └─ README.md
    ├─ .gitignore
    ├─ data.py
    ├─ dataset.py
    ├─ log.py
    └─ train.py
```

## ✏️ Evaluation

- Top-1 accuracy: 43%
- Top-5 accuracy: 80%

## 🚀 How to Start

1. Image Classification: [Link](https://github.com/boostcampaitech3/final-project-level3-cv-16/tree/main/image_classification#readme)
    
2. OCR: [Link](https://github.com/boostcampaitech3/final-project-level3-cv-16/blob/develop/README.md)
    
3. Object Detection (yolov5): [Link]()

4. Metric learning: [Link]()

## 🔎 Future Research

- 모델의 정확도 향상 및 inference time 단축
- Mobile Application 제작
- 실용성 향상
- OCR 적용

## 📎 Appendix

📄 [Experiments & Submission Report](https://www.notion.so/W18-21-Product-Serving-Project-Team-Medic-c09ea15ac67948d08fe4460194f773a8)

## 📜 Reference

- ePillID Dataset: A Low-Shot Fine-Grained Benchmark for Pill Identification ([Link](https://arxiv.org/pdf/2005.14288.pdf))
- YOLACT: Real-time Instance Segmentation ([Link](https://arxiv.org/abs/1904.02689))
- How to make deep-text-recognition-benchmark model to recognize both Korean and English ([Link](https://ropiens.tistory.com/35))
- OCR Model ([Link](https://github.com/clovaai/deep-text-recognition-benchmark))
- Jaccard Similarity ([Link](https://newscatcherapi.com/blog/ultimate-guide-to-text-similarity-with-python))
- Text-Recognition Model ([Link](https://github.com/clovaai/CRAFT-pytorch))
- Background-Removal program ([Link](https://github.com/brilam/remove-bg))
- Object Detection model YOLOv5 ([Link](https://github.com/ultralytics/yolov5))
- timm ([Link](https://github.com/rwightman/pytorch-image-models))
