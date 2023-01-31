# level3_productserving-level3-nlp-01

- [model link (on google drive)](https://drive.google.com/drive/folders/1peLB2-ngf8pYgyrgf545q1Ml-ReBHLiL?usp=sharing)
- model_path = f'/opt/ml/project_models/{model_name}'

# backend settings

```bash
apt install ffmpeg
apt-get install libsndfile1-dev
apt install default-jdk

apt-get install openjdk-8-jdk python3-dev 
bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)
```

# frontend settings

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

#### usage
웹 서버를 실행시킵니다. (not build)
```bash
$ npm start
```


