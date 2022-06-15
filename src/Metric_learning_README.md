# 🤔 How to start?

1. Clone the repository, bring code and dataset(ePillID_data_v1.0)

```cpp
https://github.com/usuyama/ePillID-benchmark
```

2. Using python train_cv.py, it is possible to select among resnet18,50,101,152 and train

```
!python train_nocv.py \
--appearance_network resnet18 or resnet50 or resnet101 or resnet152 \
--pooling CBP \
--max_epochs 10 \
--data_root_dir \
../../ePillID_data
 
embedding_model = multihead_model.embedding_model
#multi-head trainer에서 feature extractor만 저장
torch.save(embedding_model.load_state_dict()) 
```

3. Since there are two src folders in the repository, an error occurs when accessed from outside, so rename the outer src to epillid_src
---
4. trained model (resnet101 + CBP)

```cpp
https://drive.google.com/file/d/1-mdX3v3qfFSOdvtH4tS8MLy_BvlOjdeC/view
```
