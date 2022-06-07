from typing import List
from fastapi import FastAPI, File, UploadFile,Response
from pydantic import BaseModel
from ePillID_benchmark.epillid_src.models.embedding_model import EmbeddingModel
import torch
from torchvision.transforms import transforms
import uvicorn
import numpy as np
import cv2
from rembg import remove
import pymysql
from PIL import Image
from io import BytesIO
import pandas as pd
import os

conn = None
cursor = None


class ResponseItem(BaseModel):
    name: str
    shape: str
    image_url: str
    score: float


class ItemOut(BaseModel):
    items: List[ResponseItem] = []

res_mean = [0.485, 0.456, 0.406]
res_std = [0.229, 0.224, 0.225]

torch_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(res_mean, res_std)
])

app = FastAPI()
df = None
model = EmbeddingModel('resnet50')
def subimage(image, rect):
    theta = rect[2]-90
    center = (int(rect[0][0]),int(rect[0][1]))
    height = int(rect[1][0])
    width = int(rect[1][1])
    theta *= 3.14159 / 180 # convert to rad
    v_x = (np.cos(theta), np.sin(theta))
    v_y = (-np.sin(theta), np.cos(theta))
    s_x = center[0] - v_x[0] * ((width-1) / 2) - v_y[0] * ((height-1) / 2)
    s_y = center[1] - v_x[1] * ((width-1) / 2) - v_y[1] * ((height-1) / 2)

    mapping = np.array([[v_x[0],v_y[0], s_x],
                        [v_x[1],v_y[1], s_y]])

    cropped = cv2.warpAffine(image,mapping,(width, height),flags=cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_REPLICATE)
    if cropped.shape[0] > cropped.shape[1]:
        cropped = cv2.rotate(cropped,cv2.ROTATE_90_COUNTERCLOCKWISE)
    cropped = cv2.resize(cropped,(224,int(224*cropped.shape[0]/cropped.shape[1])))
    zero_image = np.zeros((224,224,3),dtype=np.uint8)
    zero_image[112-cropped.shape[0]//2 : 112 +(cropped.shape[0]-cropped.shape[0]//2),:] = cropped
    return zero_image

def segment_image(img_arr):
    if img_arr.shape[-1] == 4:
        img_arr = cv2.cvtColor(img_arr,cv2.COLOR_RGBA2RGB)
    imgray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(imgray, 10,255,cv2.THRESH_BINARY)

    # 컨튜어 찾기
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contr = contours[0]
    rect = cv2.minAreaRect(contr)
    return subimage(img_arr,rect)

@app.on_event("startup")
def startup_event():
    global model
    global df
    df = pd.read_csv('OpenData_PotOpenTabletIdntfcC20220601.csv')
    model.load_state_dict(torch.load('ePillID_benchmark/pill_encoder.pt'))
    model = model.eval()
    print('load weight done!')
    #global conn
    #global cursor
    #conn = pymysql.connect(
    #    user="sangyoon", passwd="1234", host="localhost", db="pills", charset="utf8"
    #)
    #cursor = conn.cursor()


@app.on_event("shutdown")
def shutdown_event():
    pass
    #conn.commit()
    #conn.close()

@app.post("/image_segment/")
async def query(file : UploadFile = File(...)):
    img = Image.open(BytesIO(await file.read()))
    img = remove(img)
    arr = np.array(img)
    seg = segment_image(arr)
    
    ret = Image.fromarray(seg)
    with BytesIO() as buf:
        ret.save(buf, format='PNG')
        im_bytes = buf.getvalue()
    headers = {'Content-Disposition': 'inline; filename="test.png"'}
    return Response(im_bytes, headers=headers, media_type='image/png')

def cosine_distance(a,b):
    dis = np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
    return dis

@app.post("/image_query/")
async def query(files: List[UploadFile] = File(...)):

    feats =[]
    feat_dir = 'ePillID_benchmark/features'
    for file in files:
        imgarr = np.array(Image.open(BytesIO(await file.read())))
        feats.append(model(torch_transform(imgarr).unsqueeze(0)).squeeze().detach().numpy())
    items = ItemOut()
    l = []
    for index, row in df.iterrows():
        feat_item = np.load(os.path.join(feat_dir,str(row['품목일련번호']) + '.npy'))
        l.append((max(cosine_distance(feats[0],feat_item[1])+cosine_distance(feats[1],feat_item[0]),
                    cosine_distance(feats[0],feat_item[0])+cosine_distance(feats[1],feat_item[1]),
        )/2,row['큰제품이미지'],row['품목명'],row['분류명']))
    l.sort(reverse = True)
    print(l[:10])
    for i in range(10):
        items.items.append(
            {"name": l[i][2], "image_url": l[i][1], "usage": l[i][3], "score": l[i][0]}
        )
    return items


if __name__ == "__main__":
    uvicorn.run("listen:app", host="127.0.0.1", port=8080)
