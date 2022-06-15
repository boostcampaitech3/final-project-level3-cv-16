# 🤔 How to start?

1. Clone to CRAFT-pytroch (© ClovaAI)
    
    ```bash
    git clone https://github.com/clovaai/CRAFT-pytorch.git
    pip install -r requirements.txt
    ```
    
2. Clone to deep-text-recognition-benchmark (© ClovaAI)
    
    ```bash
    git clone https://github.com/clovaai/deep-text-recognition-benchmark.git
    pip install lmdb pillow torchvision nltk natsort
    ```
    
3. [https://drive.google.com/file/d/1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ/view](https://drive.google.com/file/d/1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ/view) 에서 pth file을 다운 받습니다.
4. Clone한 CRAFT-pytroch 폴더 내에 위에서 다운받은 pth file을 저장합니다.
5. 하단 코드를 입력하여 CRAFT-pytorch를 실행합니다.
    
    ```bash
    python test.py --trained_model=[weightfile] --test_folder=[folder path to test images]
    ```
    
6. 하단 코드를 입력하여 text부분만 추출합니다.
    
    ```bash
    python ocr_text_crop.py
    ```
    
7. [https://www.dropbox.com/sh/j3xmli4di1zuv3s/AAArdcPgz7UFxIHUuKNOeKv_a?dl=0](https://www.dropbox.com/sh/j3xmli4di1zuv3s/AAArdcPgz7UFxIHUuKNOeKv_a?dl=0) 에서 TPS-ResNet-BiLSTM-Attn-case-sensitive.pth file을 다운로드 받습니다.
8. Clone 한 deep-text-recognition-benchmark 폴더 내에 saved_model 폴더를 생성 후 위에서 다운받은 pth file을 해당 디렉토리에 넣습니다.
9. 하단 코드를 입력하여 deep-text-recognition-benchmark를 실행합니다.
    
    ```bash
    python demo.py --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --image_folder demo_image/ --saved_model TPS-ResNet-BiLSTM-Attn.pth
    ```
    

# 자신만의 데이터 셋으로 학습하거나, 영어가 아닌 데이터 셋을 학습해야 하는 경우

1. Create your own lmdb dataset

```
pip install fire

# Please enter the path and file name according to the situation.
python create_lmdb_dataset.py --inputPath data/ --gtFile data/gt.txt --outputPath result/
```

1. The structure of the data folder should be as below.
    
    ```
    data
    ├── gt.txt
    └── test
        ├── word_1.png
        ├── word_2.png
        ├── word_3.png
        └── ...
    ```
    
2. At this time, gt.txt should be {imagepath}\t{label}\n
For example
    
    ```
    test/word_1.png Tiredness
    test/word_2.png kills
    test/word_3.png A
    ...
    ```
    
3. Modify —select_data, —batch_ratio, and opt.character (Note the following [Link](https://ropiens.tistory.com/35) or train_KOR_ENG_data.py)
    
``` 
parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz가각간갇갈감갑값갓강갖같갚갛개객걀걔거걱건걷걸검겁것겉게겨격겪견결겹경곁계고곡곤곧골곰곱곳공과관광괜괴굉교구국군굳굴굵굶굽궁권귀귓규균귤그극근글긁금급긋긍기긴길김깅깊까깍깎깐깔깜깝깡깥깨꺼꺾껌껍껏껑께껴꼬꼭꼴꼼꼽꽂꽃꽉꽤꾸꾼꿀꿈뀌끄끈끊끌끓끔끗끝끼낌나낙낚난날낡남납낫낭낮낯낱낳내냄냇냉냐냥너넉넌널넓넘넣네넥넷녀녁년념녕노녹논놀놈농높놓놔뇌뇨누눈눕뉘뉴늄느늑는늘늙능늦늬니닐님다닥닦단닫달닭닮담답닷당닿대댁댐댓더덕던덜덟덤덥덧덩덮데델도독돈돌돕돗동돼되된두둑둘둠둡둥뒤뒷드득든듣들듬듭듯등디딩딪따딱딴딸땀땅때땜떠떡떤떨떻떼또똑뚜뚫뚱뛰뜨뜩뜯뜰뜻띄라락란람랍랑랗래랜램랫략량러럭런럴럼럽럿렁렇레렉렌려력련렬렵령례로록론롬롭롯료루룩룹룻뤄류륙률륭르른름릇릎리릭린림립릿링마막만많말맑맘맙맛망맞맡맣매맥맨맵맺머먹먼멀멈멋멍멎메멘멩며면멸명몇모목몬몰몸몹못몽묘무묵묶문묻물뭄뭇뭐뭘뭣므미민믿밀밉밌및밑바박밖반받발밝밟밤밥방밭배백뱀뱃뱉버번벌범법벗베벤벨벼벽변별볍병볕보복볶본볼봄봇봉뵈뵙부북분불붉붐붓붕붙뷰브븐블비빌빔빗빚빛빠빡빨빵빼뺏뺨뻐뻔뻗뼈뼉뽑뿌뿐쁘쁨사삭산살삶삼삿상새색샌생샤서석섞선설섬섭섯성세섹센셈셋셔션소속손솔솜솟송솥쇄쇠쇼수숙순숟술숨숫숭숲쉬쉰쉽슈스슨슬슴습슷승시식신싣실싫심십싯싱싶싸싹싼쌀쌍쌓써썩썰썹쎄쏘쏟쑤쓰쓴쓸씀씌씨씩씬씹씻아악안앉않알앓암압앗앙앞애액앨야약얀얄얇양얕얗얘어억언얹얻얼엄업없엇엉엊엌엎에엔엘여역연열엷염엽엿영옆예옛오옥온올옮옳옷옹와완왕왜왠외왼요욕용우욱운울움웃웅워원월웨웬위윗유육율으윽은을음응의이익인일읽잃임입잇있잊잎자작잔잖잘잠잡잣장잦재쟁쟤저적전절젊점접젓정젖제젠젯져조족존졸좀좁종좋좌죄주죽준줄줌줍중쥐즈즉즌즐즘증지직진질짐집짓징짙짚짜짝짧째쨌쩌쩍쩐쩔쩜쪽쫓쭈쭉찌찍찢차착찬찮찰참찻창찾채책챔챙처척천철첩첫청체쳐초촉촌촛총촬최추축춘출춤춥춧충취츠측츰층치칙친칠침칫칭카칸칼캄캐캠커컨컬컴컵컷케켓켜코콘콜콤콩쾌쿄쿠퀴크큰클큼키킬타탁탄탈탑탓탕태택탤터턱턴털텅테텍텔템토톤톨톱통퇴투툴툼퉁튀튜트특튼튿틀틈티틱팀팅파팎판팔팝패팩팬퍼퍽페펜펴편펼평폐포폭폰표푸푹풀품풍퓨프플픔피픽필핏핑하학한할함합항해핵핸햄햇행향허헌험헤헬혀현혈협형혜호혹혼홀홈홉홍화확환활황회획횟횡효후훈훌훔훨휘휴흉흐흑흔흘흙흡흥흩희흰히힘-+/.&', help='character label')
```

## ✏️ text-recognition 모델을 의약품안전나라 데이터로 평가하기

해당 프로세스는 ipynb 파일에 정리되어 있습니다.

Check out this [link](https://github.com/boostcampaitech3/final-project-level3-cv-16/tree/develop/ocr/text_recognition_evaluation#readme)!
