# STT에서 csv 파일 전달
# csv 파일 합쳐서 -> .기준으로 문장 분리 후 -> 문단 분리
# 문단 분리한 내용으로 요약문 생성 -> 후처리

# TODO: dataloader 만들기 -> dataloader.py
    # TODO: csv 파일에서 후처리된 내용 모두 합쳐서 하나의 str로 만든 뒤, 온점 기준으로 나누어 문장 세트 구성 - input : csv -> output : 문장 세트 =>> finish
#TODO: 전처리 class 제작 -> preprocess.py
    # TODO: 문장 분리 - input : STT 문장 세트를 하나로 합친 덩어리 -> output : 문장 세트 (문장 단위로 분리된) =>> finish
    # TODO: 문단 분리 기능 - input : 문장 세트 -> output : 분리된 문단 =>> finish
# TODO: summarization class 제작 -> summary.py 
    # input : 분리된 문단 -> output : 문단 별 요약문 list
# TODO: 요약 후처리 class 제작 -> postprocess.py
    # input : 문단 별 요약문 -> output : 후처리된 요약문 결과
    # 후처리 내용
        # 반복 어구 제거
        # 어색한 단어 변경
        # 유사한 문장으로 묶어서 문단 생성하기 ~ 문단 분리 class 재사용
import pickle
import sys
import time
from dataloader import DataLoader
from preprocess import PreProcessor
from summary import Summarizer
from postprocess import PostProcessor


def main_test(data_path, sum_model_path, sum_model): # inference 시간 확인을 위한 임시 함수 -> 추후 삭제 예정
    start = time.time()

    # 1. STT output 파일 데이터셋 가져오기 (STT 후처리 완료된 파일)
    ## 추후 STT output 형식 바뀔 수도 있으니까 따로 분리함
    loader = DataLoader(data_path = data_path)
    stt_data = loader.load()

    # 2. 요약 전, 데이터 전처리 진행
    preprocessor = PreProcessor(data = stt_data,
                                stride = 4,
                                min_len = 100,
                                max_len = 1000)
    preprocessed = preprocessor.preprocess()
    prep_time = time.time()
    print('# of paragraph :', len(preprocessed))

    print('time for prep :', prep_time - start) # 5.22 초



    # 3. Summarization
    summarizer = Summarizer(data = preprocessed,
                            model_path = sum_model_path,
                            max_input_length = 512,
                            max_target_length = 64,
                            model_ = sum_model) # kobart or t5

    summarized_data = summarizer.summarize() # 문단 별 한 문장의 요약문 list 반환
    summary_time = time.time()

    print('# of summary :', len(summarized_data))
    print('time for summary :', summary_time - prep_time) # 14.77 초

    # 4. Postprocessing
    postprocessor = PostProcessor(data=summarized_data)
    postprocessed = postprocessor.postprocess()
    postprocess_time = time.time()
    print('# of postprocessed :', len(postprocessed))
    print('time for postprocessing :', postprocess_time - summary_time) # 48.77 초 -> 후처리에 드는 시간 상당 ... 
    return postprocessed

def main(data_path, sum_model_path, sum_model):
    # 1. STT output 파일 데이터셋 가져오기 (STT 후처리 완료된 파일)
    ## 추후 STT output 형식 바뀔 수도 있으니까 따로 분리함
    loader = DataLoader(data_path = data_path)
    stt_data = loader.load()

    # 2. 요약 전, 데이터 전처리 진행
    preprocessor = PreProcessor(data = stt_data,
                                stride = 4,
                                min_len = 300,
                                max_len = 1000)
    preprocessed = preprocessor.preprocess()


    # 3. Summarization
    summarizer = Summarizer(data = preprocessed,
                            model_path = sum_model_path,
                            max_input_length = 512,
                            max_target_length = 64,
                            model_ = sum_model) # kobart or t5

    summarized_data = summarizer.summarize() # 문단 별 한 문장의 요약문 list 반환

    # 4. Postprocessing
    postprocessor = PostProcessor(data=summarized_data)
    postprocessed = postprocessor.postprocess()
    return postprocessed

if __name__ == '__main__':
    result = main_test(data_path = '/opt/ml/stt_0126.csv',
              sum_model_path = '/opt/ml/project_models/summarization/kobart_all_preprocessed_without_news',
              sum_model = 'kobart')

    import pickle
    with open('summ_result.pickle', 'wb') as f:
        pickle.dump(result, f)






    