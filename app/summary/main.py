import pickle
from .dataloader import DataLoader
from .preprocess import PreProcessor
from .summary import Summarizer
from .postprocess import PostProcessor

def segment(model, data):
    # 1. STT output 파일 데이터셋 가져오기 (STT 후처리 완료된 파일)
    ## 추후 STT output 형식 바뀔 수도 있으니까 따로 분리함

    '''
    ## segment
    description:
    - stt 후처리된 결과를 입력받아 문단을 분리하는 작업을 수행합니다.
    
    args:
    - model : 문단 분리 시 유사도 계산을 위한 모델
        - SentenceTransformer
    - data : stt 후처리된 결과
        - list
    '''
    loader = DataLoader(data = data)
    stt_data = loader.load()

    # 2. 요약 전, 데이터 전처리 진행
    preprocessor = PreProcessor(model = model,
                                data = stt_data,
                                stride = 4,
                                min_len = 300,
                                max_len = 768)
    preprocessed = preprocessor.preprocess()
    return preprocessed

def summarize(model, tokenizer, postprocess_model, preprocessed: list, sum_model: str):
    '''
    ## summarize
    description:
    - 문단 리스트를 입력받아 각 문단별 요약 결과를 반환하는 역할을 수행합니다.
    
    args:
    - model : 요약 시 사용하는 모델
        - t5 사용시 : AutoModelForSeq2SeqLM
        - kobart 사용시 : BartForConditionalGeneration
    - tokeniaer : 요약 시 사용하는 tokenizer 모델
        - t5 사용시 : AutoTokenizer
        - kobart 사용시 : PreTrainedTokenizerFast
    - preprocessed : 요약 시 입력으로 들어오는 문단 리스트
        - list
    - sum_model : 요약 시 사용하는 모델의 종류
        - str 형태로 ['t5', 'kobart'] 중 하나
    '''
    # 3. Summarization
    summarizer = Summarizer(data = preprocessed,
                            model = model,
                            tokenizer = tokenizer, 
                            max_input_length = 512,
                            max_target_length = 64,
                            model_ = sum_model) # kobart or t5

    summarized_data = summarizer.summarize() # 문단 별 한 문장의 요약문 list 반환

    # 4. Postprocessing
    postprocessor = PostProcessor(model=postprocess_model, data=summarized_data)
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






    