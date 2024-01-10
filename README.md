# JEJURICA
성균관대학교 인공지능융합전공 캡스톤 프로젝트 ( 한진영 교수님 ) 수업에서 수행한 프로젝트입니다.  

제주도 방언 음성을 번역하여 영어 음성으로 발화가 나오도록 제작한 서비스입니다. 

## FLOW

다음은 저희 모델의 전체적인 프로세스입니다. 

각 모델을 튜닝 및 연결하여 서비스를 구성하였습니다. 
![image](https://github.com/yunseo4401/JEJURICA/assets/121151508/61c2197b-2875-4955-9c5b-feb06ccdd73f)

## 데이터 
제주도 방언 번역 모델링을 위해서는 KoBART를 fine-tuning 하였습니다. 

사용한 방언-표준어 코퍼스 데이터는 다음과 같습니다. 
   
|데이터 종류|개수|
|:-----:|:-----:|
|AI HUB|[84,385]|
|JIT DATASET|[114,626]|
|제주도 방언 사전 크롤|[33,646]|


## 성능평가

각 모델에 대한 bleu score 평가와, 최종 모델에 대한 human evaluation을 진행하였습니다. 
번역 모델에 한하여 bert score도 측정하였습니다.
|모델 |bleu score|
|:-----:|:-----:|
|whisper|[0.21]|
|kobart|[0.28]|
|mbart|[0.6366]|


## 서비스 이용하기 

```
1. git clone --
2. pip install -r requirements.txt
3. cd C:/ -- (레포를 git clone 한 공간으로)
4. python server.py
```
서버 생성에 필요한 모델 경로 파일은 다음 드라이브 링크에서 다운받으실 수 있습니다. (용량이 굉장히 큽니다. )

경로 : https://drive.google.com/file/d/10TDyacV2i5En2NMzLpsIneTY7TPetF-c/view

업로드 할 제주도 방언 음성은 반드시 ".wav" 형식이어야 합니다. 
