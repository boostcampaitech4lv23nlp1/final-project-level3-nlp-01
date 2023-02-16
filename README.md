# AllAtOnce : For Lecture
## Members
|강혜빈|권현정|백인진|이용우|이준원|
|:--:|:--:|:--:|:--:|:--:|
|<img width="100" alt="에브리타임" src="https://user-images.githubusercontent.com/37149278/216918705-56e2f4d6-bc4f-482a-b9fd-190ca865d0e5.png">|<img width="100" alt="에브리타임" src="https://user-images.githubusercontent.com/37149278/216918785-3bc90fc4-e4b8-43f4-bd61-d797cf87e344.png">|<img width="100" alt="에브리타임" src="https://user-images.githubusercontent.com/37149278/216919234-e9cc433c-f464-4a4b-8601-cffa668b22b2.png">|<img width="100" alt="에브리타임" src="https://user-images.githubusercontent.com/37149278/216919814-f6ff7c2f-90ea-489c-b19a-a29fca8f9861.png">|<img width="100" alt="에브리타임" src="https://user-images.githubusercontent.com/37149278/216919925-1ab02487-e7a5-4995-8d22-1253bbcae550.png">|
|Question <br> Generation|Answer <br> Extraction|Summarization, <br> FastAPI|STT, <br> React|STT, <br> STT PostProcessing|
|[@hyeb](https://github.com/hyeb)|[@malinmalin2](https://github.com/malinmalin2)|[@eenzeenee](https://github.com/eenzeenee)|[@wooy0ng](https://github.com/wooy0ng)|[@jun9603](https://github.com/jun9603)


<br><br><br>

## Introduction
### 프로젝트 배경
- 대면 수업 확대로 기존 녹화 강의에서 실시간 대면 강의로 변화함에 따라 학습에 어려움을 겪는 학생들이 많아짐.

![교육부](https://user-images.githubusercontent.com/37149278/216916559-523f6fe2-18ae-4dec-8281-0fb0d0ee9c3f.png)

<img width="500" alt="에브리타임" src="https://user-images.githubusercontent.com/37149278/216916495-addec7cf-3301-4ff0-8839-b6831b0cc426.png">



<br><br><br>

### 프로젝트 목표
- 강의 녹음본을 활용하여 요약본과 예상 질문 및 답안을 생성하여 효율적인 학습을 돕는다.

<br><br>

### 프로젝트 목표 대상
- 강의 수강 시, 주요 내용을 놓쳐 강의 녹음본을 통해 복습하고자 하는 학생
- 대면 강의 시, 필기 중 주요 내용에 대한 체계적인 정리가 어려운 학생
- 시험 전, 예상질문과 답안을 통해 공부를 하고자 하는 학생

<br><br>

## Service Flow

![pipeline](https://user-images.githubusercontent.com/37149278/216917400-b88ac143-0925-4921-9d86-d44181bcd4c5.png)

<br><br><br>

## Installization

- [model link (on google drive)](https://drive.google.com/drive/folders/1peLB2-ngf8pYgyrgf545q1Ml-ReBHLiL?usp=sharing)
- model_path = f'/opt/ml/project_models/{model_name}'

<br><br>

### backend settings

```bash
apt install ffmpeg
apt-get install libsndfile1-dev
apt install default-jdk

apt-get install openjdk-8-jdk python3-dev 
bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)
```

<br><br>

### frontend settings

아래와 같이 명령어 설치 후, terminal을 재시작합니다.
```bash
apt install nodejs
apt install npm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.2/install.sh | bash
```

nvm을 사용하여 node.js 버전 업데이트를 합니다.
```bash
nvm install 17
```

react-native를 설치합니다.
```bash
npm install -g create-react-app
```

package.json파일이 위치하는 곳에서 아래의 명령어를 입력하여 실행에 필요한 모듈을 설치합니다.
```bash
$ npm install
```

<br><br>

### usage
웹 서버를 실행시킵니다. (not build)
```bash
$ npm start
```

<br><br><br>

## Demo

![데모 시연 (2)](https://user-images.githubusercontent.com/37149278/217130400-f3f790ab-0d2d-45e3-a7a5-e66b2ab254a6.gif)


<br><br><br>


## Structure

```bash
|-- Makefile
|-- README.md
|-- app
|   |-- STT
|   |   |-- __init__.py
|   |   |-- data
|   |   |   |-- __init__.py
|   |   |   |-- audio.py
|   |   |   |-- conf.yaml
|   |   |   |-- make_dataset.py
|   |   |   `-- utils
|   |   |       |-- __init__.py
|   |   |       |-- custom_split.py
|   |   |       |-- delete_loop.py
|   |   |       `-- output_to_dataframe.py
|   |   |-- inference
|   |   |   |-- __init__.py
|   |   |   `-- inference.py
|   |   `-- setup.py
|   |-- __init__.py
|   |-- __main__.py
|   |-- client.py
|   |-- keyword_extraction
|   |   |-- __init__.py
|   |   |-- data_utils
|   |   |   |-- __init__.py
|   |   |   |-- ner_dataset.py
|   |   |   |-- pad_sequence.py
|   |   |   |-- utils.py
|   |   |   `-- vocab_tokenizer.py
|   |   |-- filtering.py
|   |   |-- keybert_model.py
|   |   |-- main.py
|   |   |-- ner_config
|   |   |   |-- config.json
|   |   |   |-- ner_to_index.json
|   |   |   |-- net.py
|   |   |   |-- pytorch_kobert.py
|   |   |   `-- utils.py
|   |   |-- stopwords.txt
|   |   `-- vocab.pkl
|   |-- question_generation
|   |   |-- __init__.py
|   |   |-- kobart_qg.py
|   |   |-- main.py
|   |   |-- qg_filtering.py
|   |   `-- t5_pipeline.py
|   |-- server.py
|   |-- stt_postprocessing
|   |   |-- __init__.py
|   |   |-- dataloader.py
|   |   |-- inference.py
|   |   |-- main.py
|   |   `-- split_data.py
|   |-- summary
|   |   |-- __init__.py
|   |   |-- dataloader.py
|   |   |-- main.py
|   |   |-- postprocess.py
|   |   |-- preprocess.py
|   |   `-- summary.py
|   `-- utils
|       |-- keyword_model_init.py
|       |-- qg_model_init.py
|       |-- stt_model_init.py
|       `-- summary_model_init.py
|-- debug.py
|-- frontend
|   |-- README.md
|   |-- package-lock.json
|   |-- package.json
|   |-- public
|   |   |-- favicon.ico
|   |   |-- index.html
|   |   |-- logo192.png
|   |   |-- logo512.png
|   |   |-- manifest.json
|   |   `-- robots.txt
|   `-- src
|       |-- App.css
|       |-- App.js
|       |-- App.test.js
|       |-- images
|       |   `-- background.png
|       |-- index.css
|       |-- index.js
|       |-- logo.svg
|       |-- questionGenerationPage.js
|       |-- reportWebVitals.js
|       |-- setupTests.js
|       |-- sttPage.js
|       `-- summarizationPage.js
|-- poetry.lock
`-- pyproject.toml
```

<br><br><br>

## Review

[Wrap-up Report](https://docs.google.com/document/d/1Ars4xIwXqjSFoMxxC03JUyKm_UsWZ1RrXQmJZqgxbiw/edit?usp=sharing)

<!-- [Wrap-up Notion]() -->

[Presentation](https://docs.google.com/presentation/d/1w4aHEkJTYKRBfi6exVl3DvRwXsg2ipxe/edit?usp=sharing&ouid=110238381982950597659&rtpof=true&sd=true)

<br><br><br>

## Reference
**STT**

- **Character error rates(CER) : [https://github.com/hyeonsangjeon/computing-Korean-STT-error-rates](https://github.com/hyeonsangjeon/computing-Korean-STT-error-rates)**
- **ESPNet : [https://github.com/espnet/espnet](https://github.com/espnet/espnet)**
- **Whisper : [https://arxiv.org/pdf/2212.04356.pdf](https://arxiv.org/pdf/2212.04356.pdf)**

**Summarization**

- **KoBART : https://github.com/seujung/KoBART-summarization / [https://huggingface.co/docs/transformers/model_doc/bart](https://huggingface.co/docs/transformers/model_doc/bart)**
- **T5 : https://github.com/google-research/multilingual-t5 / [https://github.com/google-research/text-to-text-transfer-transformer](https://github.com/google-research/text-to-text-transfer-transformer)**
- **반복 문자열 제거 : [https://velog.io/@likelasttime/%ED%8C%8C%EC%9D%B4%EC%8D%AC-4873.-%EB%B0%98%EB%B3%B5%EB%AC%B8%EC%9E%90-%EC%A7%80%EC%9A%B0%EA%B8%B0](https://velog.io/@likelasttime/%ED%8C%8C%EC%9D%B4%EC%8D%AC-4873.-%EB%B0%98%EB%B3%B5%EB%AC%B8%EC%9E%90-%EC%A7%80%EC%9A%B0%EA%B8%B0)**

**Answer Extraction**

- **KeyBERT : [https://github.com/MaartenGr/KeyBERT](https://github.com/MaartenGr/KeyBERT)**
- **NER : [https://github.com/eagle705/pytorch-bert-crf-ner](https://github.com/eagle705/pytorch-bert-crf-ner)**

**Question Generation**

- **KoBART : [https://github.com/Seoneun/KoBART-Question-Generation](https://github.com/Seoneun/KoBART-Question-Generation)**
- **T5 : [https://github.com/patil-suraj/question_generation#project-details](https://github.com/patil-suraj/question_generation#project-details) / [https://huggingface.co/paust/pko-t5-base](https://huggingface.co/paust/pko-t5-base)**

**FastAPI : [https://fastapi.tiangolo.com/ko/](https://fastapi.tiangolo.com/ko/)**