import pickle
from .dataloader import DataLoader
from .preprocess import PreProcessor
from .summary import Summarizer
from .postprocess import PostProcessor

def segment(model, data):
    # 1. STT output 파일 데이터셋 가져오기 (STT 후처리 완료된 파일)
    ## 추후 STT output 형식 바뀔 수도 있으니까 따로 분리함
    loader = DataLoader(data = data)
    stt_data = loader.load()

    # 2. 요약 전, 데이터 전처리 진행
    preprocessor = PreProcessor(model = model,
                                data = stt_data,
                                stride = 4,
                                min_len = 300,
                                max_len = 1000)
    preprocessed = preprocessor.preprocess()
    return preprocessed

def summarize(model, tokenizer, preprocessed: list, sum_model: str):
    # 3. Summarization
    summarizer = Summarizer(data = preprocessed,
                            model = model,
                            tokenizer = tokenizer, 
                            max_input_length = 512,
                            max_target_length = 64,
                            model_ = sum_model) # kobart or t5

    summarized_data = summarizer.summarize() # 문단 별 한 문장의 요약문 list 반환

    # 4. Postprocessing
    postprocessor = PostProcessor(data=summarized_data)
    postprocessed = postprocessor.postprocess()
    return postprocessed

if __name__ == '__main__':
    prerpocessed = summ_preprocess(data = '/opt/ml/Segmentation/data/dataset_0118.csv')
    result = summarize(data = prerpocessed,
              sum_model_path = '/opt/ml/KoBART-summarization/MODEL/binary_models/kobart_all_preprocessed_without_news',
              sum_model = 'kobart')

    import pickle
    with open('summ_result.pickle', 'wb') as f:
        pickle.dump(result, f)






    