import streamlit as st
import io
from PIL import Image
import requests
from pathlib import Path
import base64

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded





st.set_page_config(
        page_title="너의 알약이 보여?", layout="centered", page_icon="pill"
    )




def main():
    st.title('너의 알약이 보여')
    st.sidebar.title("사용법")
    st.sidebar.subheader("1. 알아보고 싶은 알약의 앞면과 뒷면이 잘 보이도록 사진을 각각 한 장씩 찍습니다.")
    st.sidebar.subheader("2. 찍은 알약의 앞면 사진을 상단에, 뒷면 사진을 하단에 업로드합니다.")
    st.sidebar.subheader("3. 알약 영역이 잘 분리되었는지 확인해줍니다.")
    st.sidebar.subheader("4. 사진이 잘 분리되었다면 예측 시작 버튼을 누릅니다.")
    st.sidebar.subheader("5. 업로드한 사진과 유사한 알약을 선택한 갯수까지 웹페이지 상으로 보여줍니다.")

    st.sidebar.title("주의사항")
    st.sidebar.subheader("1. 최대한 빛번짐이 없는 밝은 곳에서 찍어주세요.")
    st.sidebar.subheader("2. 사진에 알약이 하나만 찍히도록 해주세요.")
    st.sidebar.subheader("3. 만약 5개 중에 없다면 10개 표시를 체크하고 다시 검색해주세요.")
    st.sidebar.subheader("4. 전체 검색을 사용하면 시간은 느리지만 더 빠른 결과를 얻을 수 있습니다..")

    col1,col2= st.columns(2)
    uploaded_file_front = col1.file_uploader(
        "알약 앞면 사진을 올려주세요.", type=["jpg", "jpeg", "png"]
    )
    uploaded_file_back = col2.file_uploader(
        "알약 뒷면 사진을 올려주세요.", type=["jpg", "jpeg", "png"]
    )
    ten = st.checkbox('10개 검색')
    from_all = st.checkbox('전체 검색')
    if uploaded_file_front and uploaded_file_back:
        if st.button("예측 시작"):
            files = [
                ("files", uploaded_file_front.getvalue()),
                ("files", uploaded_file_back.getvalue()),
            ]
            with st.spinner("잠시만 기다려주세요"):
                if from_all:
                    res = requests.post("http://127.0.0.1:8080/image_query/", files=files, params = {'all' : True})
                else:
                    res = requests.post("http://127.0.0.1:8080/image_query/", files=files, params = {'all' : False})
            outputs = res.json()
            k = 10 if ten else 5
            if outputs["items"][0]['valid']:
                outn = min(len(outputs["items"]), k)
                outputs = outputs["items"]
   
                for i in range(min(outn,5)):
                    output = outputs[i]
                    response = requests.get(output['image_url'])
                    img = Image.open(io.BytesIO(response.content))
                    st.write(f"{i+1}. {output['name']}")
                    st.image(img)
                    #cols[i].write(f"효능: {output['usage']}")
                    st.write(f"유사도: {output['score']}")
                if outn >5:
                    cols2 = st.columns(outn-5)
                    for i in range(outn-5):
                        output = outputs[i+5]
                        st.write(f"{i+6}. {output['name']}")
                        response = requests.get(output['image_url'])
                        img = Image.open(io.BytesIO(response.content))
                        st.image(img)
                    # cols2[i].write(f"효능: {output['usage']}")
                    st.write(f"유사도: {output['score']}")
            else:
                output = outputs["items"][0]
                if outputs['msg1'] == 0:
                    st.write('앞면 사진에서 알약이 검출되지 않았습니다.')
                elif outputs['msg1'] == 2:
                    st.write('앞면 사진에서 2개 이상의 알약이 검출되었습니다.')
                if outputs['msg2'] == 0:
                    st.write('뒷면 사진에서 알약이 검출되지 않았습니다.')
                elif outputs['msg2'] == 2:
                    st.write('뒷면 사진에서 2개 이상의 알약이 검출되었습니다.')
if __name__ == "__main__":
    main()
