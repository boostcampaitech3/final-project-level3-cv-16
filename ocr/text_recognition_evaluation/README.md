# Evaluation of deep-text-recognition-benchmark on Pill Data

### Directory Structure
- [text_recog_analysis.ipynb](https://github.com/boostcampaitech3/final-project-level3-cv-16/blob/develop/ocr/text_recognition_evaluation/text_recog_analysis.ipynb)
  - This ipynb file compares results of [deep-text-recognition-benchmark model](https://github.com/boostcampaitech3/final-project-level3-cv-16/tree/develop/ocr/deep-text-recognition-benchmark) tested with labeled pill image dataset created with [CRAFT model](https://github.com/boostcampaitech3/final-project-level3-cv-16/tree/develop/ocr/CRAFT-pytorch).
- log_results
  - `gt.txt`: Ground truth data label
    - Dataset image created with CRAFT, then labeled within CV-16 Team Medic
  - `log_demo_result_no_case.txt`: label predicted with TPS-ResNet-BiLSTM-Attn.pth
  - `log_demo_result_new_case_sensitive.txt`: label predicted with case_sensitive version of TRBA
