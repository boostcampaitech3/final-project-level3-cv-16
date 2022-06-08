# How to download and preprocess pill data

1. Download pill data
```bash
python download_pill_data.py
```

2. Remove background from the pill data
reference: https://github.com/danielgatis/rembg
```bash
git clone https://github.com/danielgatis/rembg.git
cd rembg
conda create --name segmentation
conda activate segmentation
pip install rembg[gpu]
pip install -r requirements
pip install -r requirements-gpu.txt
rembg p path/to/input_folder path/to/output_folder
# example: rembg p ../data/raw_data ../data/background_removed_data
```

3. Detect pill & order all the pills in the same direction for training
```bash
python normalize_pill_data.py
```
