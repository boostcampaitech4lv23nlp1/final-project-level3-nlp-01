import pickle
import time
from .dataloader import DataLoader
from .preprocess import PreProcessor
from .summary import Summarizer
from .postprocess import PostProcessor


def main_test(data, sum_model_path, sum_model): # inference 시간 확인을 위한 임시 함수 -> 추후 삭제 예정
    start = time.time()

    # 1. STT output 파일 데이터셋 가져오기 (STT 후처리 완료된 파일)
    ## 추후 STT output 형식 바뀔 수도 있으니까 따로 분리함
    loader = DataLoader(data = data)
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

def main(data, sum_model_path, sum_model):
    # 1. STT output 파일 데이터셋 가져오기 (STT 후처리 완료된 파일)
    ## 추후 STT output 형식 바뀔 수도 있으니까 따로 분리함
    loader = DataLoader(data = data)
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
    result = main_test(data_path = '/opt/ml/Segmentation/data/dataset_0118.csv',
              sum_model_path = '/opt/ml/KoBART-summarization/MODEL/binary_models/kobart_all_preprocessed_without_news',
              sum_model = 'kobart')

    import pickle
    with open('summ_result.pickle', 'wb') as f:
        pickle.dump(result, f)






    