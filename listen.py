from typing import List
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import uvicorn
import numpy as np
import base64
import pymysql

conn = None
cursor = None


class ResponseItem(BaseModel):
    name: str
    shape: str
    image_url: str
    score: float


class ItemOut(BaseModel):
    items: List[ResponseItem] = []


app = FastAPI()


@app.on_event("startup")
def startup_event():
    global conn
    global cursor
    conn = pymysql.connect(
        user="sangyoon", passwd="1234", host="localhost", db="pills", charset="utf8"
    )
    cursor = conn.cursor()


@app.on_event("shutdown")
def shutdown_event():
    conn.commit()
    conn.close()


@app.post("/image_query/")
def query(files: List[UploadFile] = File(...)):

    sql = """
        SELECT  품목명,큰제품이미지,성상, feature
        from pills_table
        where 색상앞 = '연두'
        and 의약품제형 = '원형'
    """

    cursor.execute(sql)
    res = cursor.fetchall()
    items = ItemOut()
    random_feature = np.random.rand(1024)
    for i in range(min(5, len(res))):
        name = res[i][0]
        image_url = res[i][1]
        shape = res[i][2]
        feat = np.frombuffer(
            base64.b64decode(res[i][3].encode("utf-8")), dtype=np.float64
        )
        score = np.dot(feat, random_feature) / (
            (np.dot(feat, feat) * np.dot(random_feature, random_feature)) ** 0.5
        )
        items.items.append(
            {"name": name, "image_url": image_url, "shape": shape, "score": score}
        )
    return items


if __name__ == "__main__":
    uvicorn.run("listen:app", host="127.0.0.1", port=8080)
