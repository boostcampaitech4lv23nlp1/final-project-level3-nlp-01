import time
import json
import argparse
import nltk
nltk.download('punkt')
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def model_init(args):
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, stride=128)
    
    model.config.max_length = args.max_target_length
    tokenizer.model_max_length = args.max_target_length
    return model, tokenizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparameter for Inference')
    parser.add_argument('--model_path', default='models/', type=str,
                                                help='model path for inference')
    parser.add_argument('--max_input_length', default=1024, type=int,
                                                help='max input length for summarization')
    parser.add_argument('--max_target_length', default=256, type=int,
                                                help='max target length for summarization')
    parser.add_argument('--prefix', default='summarize: ', type=str,
                                                help='inference input prefix')
    parser.add_argument('--file_path', default='./data/test.json', type=str,
                                                help='file path for inference')
    parser.add_argument('--result_path', default='result.json', type=str,
                                                help='path for saving result')
    args = parser.parse_args()


    start = time.time()

    model, tokenizer = model_init(args)

    load_model_time = time.time()

    print('loading time >>', load_model_time - start)

    if args.file_path:
        with open(args.file_path, 'r') as f:
            data = json.load(f)
            
        all_result = []
        for i in range(len(data)):
            tmp_dict = dict()
            sample = data[i]['source']
            inputs = [args.prefix + sample]
            inputs = tokenizer(inputs, max_length=args.max_input_length, truncation=True, return_tensors="pt")
            output = model.generate(**inputs, num_beams=8, do_sample=True, min_length=10, max_length=args.max_target_length)
            decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
            result = nltk.sent_tokenize(decoded_output.strip())[0]
            tmp_dict['source'] = sample
            tmp_dict['result'] = result
            all_result.append(tmp_dict)

        with open(args.result_path, 'w') as f:
            json.dump(all_result, f)


    else:
        sample = """
            안녕하세요? 우리 (2학년)/(이 학년) 친구들 우리 친구들 학교에 가서 진짜 (2학년)/(이 학년) 이 되고 싶었는데 학교에 못 가고 있어서 답답하죠? 
            그래도 우리 친구들의 안전과 건강이 최우선이니까요 오늘부터 선생님이랑 매일 매일 국어 여행을 떠나보도록 해요. 
            어/ 시간이 벌써 이렇게 됐나요? 늦었어요. 늦었어요. 빨리 국어 여행을 떠나야 돼요. 
            그런데 어/ 국어여행을 떠나기 전에 우리가 준비물을 챙겨야 되겠죠? 국어 여행을 떠날 준비물, 교안을 어떻게 받을 수 있는지 선생님이 설명을 해줄게요. 
            (EBS)/(이비에스) 초등을 검색해서 들어가면요 첫화면이 이렇게 나와요. 
            자/ 그러면요 여기 (X)/(엑스) 눌러주(고요)/(구요). 저기 (동그라미)/(똥그라미) (EBS)/(이비에스) (2주)/(이 주) 라이브특강이라고 되어있죠? 
            거기를 바로 가기를 누릅니다. 자/ (누르면요)/(눌르면요). 어떻게 되냐? b/ 밑으로 내려요 내려요 내려요 쭉 내려요. 
            우리 몇 학년이죠? 아/ (2학년)/(이 학년) 이죠 (2학년)/(이 학년)의 무슨 과목? 국어. 
            이번주는 (1주)/(일 주) 차니까요 여기 교안. 다음주는 여기서 다운을 받으면 돼요. 
            이 교안을 클릭을 하면, 짜잔/. 이렇게 교재가 나옵니다 .이 교안을 (다운)/(따운)받아서 우리 국어여행을 떠날 수가 있어요. 
            그럼 우리 진짜로 국어 여행을 한번 떠나보도록 해요? 국어여행 출발. 자/ (1단원)/(일 단원) 제목이 뭔가요? 한번 찾아봐요. 
            시를 즐겨요 에요. 그냥 시를 읽어요 가 아니에요. 시를 즐겨야 돼요 즐겨야 돼. 어떻게 즐길까? 일단은 내내 시를 즐기는 방법에 대해서 공부를 할 건데요. 
            그럼 오늘은요 어떻게 즐길까요? 오늘 공부할 내용은요 시를 여러 가지 방법으로 읽기를 공부할겁니다. 
            어떻게 여러가지 방법으로 읽을까 우리 공부해 보도록 해요. 오늘의 시 나와라 짜잔/! 시가 나왔습니다 시의 제목이 뭔가요? 다툰 날이에요 다툰 날. 
            누구랑 다퉜나 동생이랑 다퉜나 언니랑 친구랑? 누구랑 다퉜는지 선생님이 시를 읽어 줄 테니까 한번 생각을 해보도록 해요."""

        inputs = [args.prefix + sample]


        inputs = tokenizer(inputs, max_length=args.max_input_length, truncation=True, return_tensors="pt")
        output = model.generate(**inputs, num_beams=8, do_sample=True, min_length=10, max_length=args.max_target_length)
        decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        result = nltk.sent_tokenize(decoded_output.strip())[0]
        
        print('RESULT >>', result)
    print('inference time >>', time.time() - load_model_time)
    print('all time >>', time.time() - start)