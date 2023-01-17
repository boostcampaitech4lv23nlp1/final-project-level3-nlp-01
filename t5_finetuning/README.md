# T5_finetuning_for_summary

한국어 데이터셋을 활용하여 T5 모델을 text summarization task를 위해 finetuning 하는 코드입니다.

## Repository 구조

```
├── README.md
├── requirements.txt
├── t5_lib.txt
├── train.py
├── utils.py
├── dataloader.py
└── infer.py
```

## Requirements
- Python 3 (tested on 3.8)
- CUDA (tested on 11.3)

`conda create -n t5 python=3.8`  
`conda install -c anaconda numpy`  
`conda install -c conda-forge transformers`  
`conda install -c conda-forge datasets`  
`conda install -c anaconda nltk`

or

`conda env create -n t5 python=3.8 -f requirements.txt`

and then,
`conda activate t5`

## Training

`python train.py`

If you want to change hyper-parameters for training,  
`python train.py --num_train_epochs 5 --train_batch_size 16 ...`

## Inference

`python infer.py`

If you want to change hyper-parameters for inference,

`python train.py --file_path ./data/test.json ...`

## Data format

In **train/val/test.json** file,

```
[{'source':'...', 'target':'...'}, {'source':'...', 'target':'...'}, ...]
```

- you can download dataset for finetuning from https://drive.google.com/drive/folders/19tGCqZ4zPLKEUpJj_83Siavmg8yP_Fv1?usp=share_link
- it is a modified version of Aihub's dataset.
    - 요약문 및 레포트 생성 데이터 : https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=582
    - 도서자료 요약 : https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=93
    - 논문자료 요약 : https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=90

After inference, in **result.json** file,

```
[{'source':'...', 'target':'...'}, {'source':'...', 'target':'...'}, ...]
```

