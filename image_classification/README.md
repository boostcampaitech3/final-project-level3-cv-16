# Image Classification

## Image Classification

1. Downloading the github repository
    
    ```bash
    git clone https://github.com/boostcampaitech3/final-project-level3-cv-16.git
    cd final-project-level3-cv-16/Image_Classification
    ```
    
2. Installing the requirements for training
    
    ```bash
    pip install -r requirements.txt
    pip install openpyxl
    ```
    
3. Download Pill Excel Data
    
    ```bash
    cd pill_excel_data
    ```
    
4. Preprocess Pill Data
    
    ```bash
    # follow the steps in /data_preprocessing
    ```
    
5. Train Pill Data for Image Classification
    
    ```bash
    python train.py
    ```
    
    ```text
    usage: train.py [-h] [--epochs EPOCHS] [--batch_size BATCH_SIZE] [--patience PATIENCE] [--learning_rate LEARNING_RATE] [--lr_decay_step LR_DECAY_STEP]
                    [--accumulation_steps ACCUMULATION_STEPS] [--train_log_interval TRAIN_LOG_INTERVAL] [--seed SEED] [--opt OPT] [--sch SCH] [--excel_file_name EXCEL_FILE_NAME]
                    [--image_file_path IMAGE_FILE_PATH] [--project_type PROJECT_TYPE] [--user_name USER_NAME] [--model_name MODEL_NAME] [--project_name PROJECT_NAME]
                    [--train_whole TRAIN_WHOLE] [--create_test_data CREATE_TEST_DATA] [--test TEST] [--delete_pill_num DELETE_PILL_NUM [DELETE_PILL_NUM ...]] [--custom_label CUSTOM_LABEL]
    
    optional arguments:
      -h, --help            show this help message and exit
      --epochs EPOCHS       number of epochs to train (default: 50)
      --batch_size BATCH_SIZE
                            input batch size for training (default: 64)
      --patience PATIENCE   early stopping (default: 10)
      --learning_rate LEARNING_RATE
                            learning rate (defalt: 0.0001)
      --lr_decay_step LR_DECAY_STEP
                            learning rate deacy step (default: 5)
      --accumulation_steps ACCUMULATION_STEPS
                            training accumulation steps (default: 2)
      --train_log_interval TRAIN_LOG_INTERVAL
                            training log interval (default: 100)
      --seed SEED           fix seed (default: 16)
      --opt OPT             optimizer (default: Adam)
      --sch SCH             scheduler (default: StepLR)
      --excel_file_name EXCEL_FILE_NAME
                            name of the pill data excel (default: ./pill_excel_data/OpenData_PotOpenTabletIdntfc20220412.xls)
      --image_file_path IMAGE_FILE_PATH
                            path to image file (default: ./data/raw_data)
      --project_type PROJECT_TYPE
                            which column to use (default: shape)
      --user_name USER_NAME
                            user name (default: YH)
      --model_name MODEL_NAME
                            timm model name (default: resnet18)
      --project_name PROJECT_NAME
                            customize project name of what difference the project has (default: None)
      --train_whole TRAIN_WHOLE
                            train without validation set (default: False)
      --create_test_data CREATE_TEST_DATA
                            create test data from training dataset (default: False)
      --test TEST           do test with custom dataset (default: False)
      --delete_pill_num DELETE_PILL_NUM [DELETE_PILL_NUM ...]
                            list of file to delete (default: [])
      --custom_label CUSTOM_LABEL
                            customize labels for training (default: False)
    ```
    
    <img width="1290" alt="Untitled (1)" src="https://user-images.githubusercontent.com/73840274/173296685-37b480d1-42fc-4815-923c-1a60e7f345c1.png">

    
6. pretrained model(resnet 50)

  [https://drive.google.com/file/d/1rDqbm3S9-0kWIhQngSuCM7VbLvmRUmbF/view?usp=sharing](https://drive.google.com/file/d/1rDqbm3S9-0kWIhQngSuCM7VbLvmRUmbF/view?usp=sharing)

## Data Preprocessing for Segmentation Tasks

```bash
# follow the steps in /kaggle_pill_data_preprocessing
```

## Data Concatenation for Test Images(Not Used Now)
> Not used due to image normalization process done in `Preprocess Pill Data`

```bash
# follow the steps in /image_concatenation
```
