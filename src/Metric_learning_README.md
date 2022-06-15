git clone [https://github.com/usuyama/ePillID-benchmark](https://github.com/usuyama/ePillID-benchmark)

에서 코드와 데이터셋(ePillID_data_v1.0) 다운로드 가능

python train_cv.py를 이용하여 resnet18,50,101,152 중 선택하여 training이 가능

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

해당 repository 내에 src 폴더가 2개가 존재해서 외부에서 접근하면 오류가 생기므로

바깥쪽의 src를 epillid_src로 이름을 바꿔줌

trained model (resnet101 + CBP)

[https://drive.google.com/file/d/1-mdX3v3qfFSOdvtH4tS8MLy_BvlOjdeC/view?usp=sharing](https://drive.google.com/file/d/1-mdX3v3qfFSOdvtH4tS8MLy_BvlOjdeC/view?usp=sharing)
