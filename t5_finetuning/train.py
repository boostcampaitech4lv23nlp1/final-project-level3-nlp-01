import argparse

import transformers
import datasets
import nltk
nltk.download('punkt')
import numpy as np
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import T5TokenizerFast, T5ForConditionalGeneration
from datasets import load_metric

from dataloader import load_data
from utils import clean_data, preprocess_data

def model_init(args):
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint, use_fast=True, stride=128)
    
    model.config.max_length = args.max_target_length
    tokenizer.model_max_length = args.max_target_length
    return model, tokenizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparameter for Training T5 for summary')
    parser.add_argument('--model_checkpoint', default='paust/pko-t5-base', type=str,
                                                help='huggingface model name to train')
    parser.add_argument('--prefix', default='summarize: ', type=str,
                                                help='inference input prefix')
    parser.add_argument('--max_input_length', default=512, type=int,
                                                help='max input length for summarization')
    parser.add_argument('--max_target_length', default=64, type=int,
                                                help='max target length for summarization')
    parser.add_argument('--use_auto_find_batch_size', default=False, type=bool,
                                                help='if you want to find batch size automatically, set True')
    parser.add_argument('--train_batch_size', default=8, type=int,
                                                help='train batch size')
    parser.add_argument('--eval_batch_size', default=8, type=int,
                                                help='eval batch size')
    parser.add_argument('--num_train_epochs', default=15, type=int,
                                                help='train epoch size')
    parser.add_argument('--lr', default=1e-5, type=int,
                                                help='learning rate for training')
    parser.add_argument('--wd', default=0.01, type=int,
                                                help='weight decay for training')
    parser.add_argument('--steps', default=30000, type=int,
                                                help='evaluation, logging, saving step for training')                                            
    parser.add_argument('--model_name', default='t5-base-korean-finetuned-for-summary', type=str,
                                                help='model name for saving')
    parser.add_argument('--base_path', default='./data/', type=str,
                                                help='dataset path')
    parser.add_argument('--model_path', default='./models', type=str,
                                                help='model path for saving')
    parser.add_argument('--predict', default=True, type=bool,
                                                help='if you want to summary some example text, set True')
    args = parser.parse_args()

    # Load datset
    dataset = load_data(args.base_path)

    # Load model & tokenizer
    model, tokenizer = model_init(args)

    # Preprocessing dataset
    dataset_cleaned = dataset.filter(lambda example: (len(example['source']) >= 200) and (len(example['target']) >= 20))
    tokenized_datasets = dataset_cleaned.map(lambda x: preprocess_data(x, tokenizer, args), batched=True)

    # Finetuning
    model_dir = f"{args.model_path}/{args.model_name}"

    if args.use_auto_find_batch_size:
        training_args = Seq2SeqTrainingArguments(
            model_dir,
            evaluation_strategy="steps", eval_steps=args.steps,
            logging_strategy="steps", logging_steps=args.steps,
            save_strategy="steps", save_steps=args.steps,
            learning_rate=args.lr,
            weight_decay=args.wd,
            auto_find_batch_size=True,
            num_train_epochs=args.num_train_epochs,
            save_total_limit=3,
            predict_with_generate=True,
            fp16=True,
            load_best_model_at_end=True,
            metric_for_best_model="rouge1",
        )
    else:
        training_args = Seq2SeqTrainingArguments(
            model_dir,
            evaluation_strategy="steps", eval_steps=args.steps,
            logging_strategy="steps", logging_steps=args.steps,
            save_strategy="steps", save_steps=args.steps,
            learning_rate=args.lr,
            weight_decay=args.wd,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            num_train_epochs=args.num_train_epochs,
            save_total_limit=3,
            predict_with_generate=True,
            fp16=True,
            load_best_model_at_end=True,
            metric_for_best_model="rouge1",
        )

    data_collator = DataCollatorForSeq2Seq(tokenizer)

    metric = load_metric("rouge")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Rouge expects a newline after each sentence
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip()))
                        for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) 
                        for label in decoded_labels]
        
        # Compute ROUGE scores
        result = metric.compute(predictions=decoded_preds, references=decoded_labels,
                                use_stemmer=True)

        # Extract ROUGE f1 scores
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        
        # Add mean generated length to metrics
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id)
                        for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)
        
        return {k: round(v, 4) for k, v in result.items()}

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
        
    # Training
    print('Start Training...')
    
    trainer.train()

    # Saving model
    print('Saving Model...')
    trainer.save_model()

    if args.predict:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

        sample = """
        안녕하세요 이반 비디오에서는 짧게 벌케 대해 이해 보도록 할게요 멀트는 마이드렉시나 잉코더 프람 트랜스포머란 뜻인데요 이름을 통해서 버트는 트랜스포머의 마이드렉시나 인코더임을 알 수가 있어요 마이드렉시나는 양방향을 의미하고 인코더는 입력값을 숫자의 형태로 바꾸는 모두를 의미하니까 버튼의 몸맥을 양방향을 위해서 숫자의 형태로 바꿔주는 딥런이 모델이다라고 알 수가 있겠습니다 트랜스포머는 2017년에
        구글에서 공개한 인코더 디코더 구조를 지인 딥러닝 모델입니다 어텐션 이제 오일균이드라는 논문을 통해서 공개화 되었고 기계본 역에서 우순한 성능을 보여준 모델이구요 방금 인코더 디코더 구조를 지낸 딥러닝 모델이라고 말씀을 드렸는데 인코더는 입력값을 양방향으로 처리하고 디코더는 왼쪽에서 오른쪽으로 입력을 단방향으로 처리한다는 큰 차이점이 있어요 월트가 양방향 인코더 형태를 취한 데는 재미난 이유가 있는
        요 그 시작은 gpt 원으로부터 시작이 됩니다 gpt 원은 2018년에 오픈의 ai에서 트랜스포머이티 코도 구조를 사용해서 만든 자연어처리 모델이에요 gpt 원은 제네티브 츄이닝으로 학습된 랭귀지 모델이 얼마나 자연어처리 능력이 우수한지 보여주는 우수한 모델입니다 기본적으로 문장을 데이터로 사용하구요 단어를 하나씩 읽어가면서 다음 단어를 예측하는 방법으로 모델이 학습이 됩니다
        이런 학습 방식은 결도에 레블링 작업이 필요가 없어서 bg도 학습이라고 말씀드릴 수 있구요 슬라이드에서 보실 수 있듯이 한 문장만 가지고도 여러 학습 데이터를 만드는 게 가능해요 문장에서 현재 위치의 단어 다음에 위치한 단어를 예측하는 방식으로 학습되기 때문에 사람이 직접 레이블링 할 필요가 없는 거죠 gbt는 이렇게 현재 위치의 단어 다음에 위치한 단어를 예측하는 방법으로 학습이 됩니다 따라서
        gpt학습시에 가장 필요한 건 엄청난 양에 데이터기가 있죠 물론 질적으로 좋은 데이터를 선별하는 노력도 엄청나게 중요합니다 인터넷상에는 텍스트가 정말 어마어마하게 많고 질 좋은 데이터를 선별하는 기술도 함께 발전하기 때문에 gpt는 앞으로도 분명히 각광받을 모델일 거에요 멀티에 재밌는 탄생 비원은 여기서 시작이 됩니다 멀티는 2018년 gpt 원에 발표 이후에 얼마 지나지 않아서
        구글이 발표하게 되는데요 구글은 참구로 트랜스포머를 만든 기업이죠 구글은 벌트 논문을 통해서 gp 티원에 트랜스포머 디코들 사용완 자연어처리 능력은 문장을 처리하는데 부족함이 있을 수 있다고 얘기해요 더블어 제리미 응답 능력은 문매 위해 능력이 상당히 중요한데 단순히 왼쪽에서 오른쪽으로 읽어나가는 방식으로는 문매 위에 약점이 있을 수 있다고 또 지적을 합니다 위에 단순히 왼쪽에서 오른쪽을 읽어나간 디코더보다
        방향으로 문맥을 이해할 수 있는 인코더를 활용한 랭귀지 모드를 벌트라는 이름으로 발표를 하게 돼요 이쯤에서 트랜스포머 작동 방식을 아주 짧게 알아보고 갈게요 입력값은 먼저 인코드 입력이 됩니다 입력값이 인코드 입력되면 각 토큰들은 포지션을 인코딩과 더해지구요 인코더는 이 값들을 행렬 계산을 통해서 한방에 어텐션 백터를 생각합니다 어텐션 백터는 토큰에 의미를 구하기 위해서 사용이 됩니다
        단어 하나만 보면 그 단어에 의미가 상당히 모호할 때가 많아요 예를 들어서 문장 전체를 보지 않고 단순히 텍스트라는 단어만 보면 텍스트가 신문 속에 있는 문장들 의미하는지 문자 메세지를 보내는 행위를 말하는지 알기가 어렵습니다 마찬가지로 메세지라는 단어만 봤을 경우에 이게 누가 말론 전환 소식인지 아니면 문자인지 헷갈리죠 여기 제가 어텐션 벡터를 이야기 위해서 아주 간략하게 단순화 시켜봤는데요 각각의 톡한들은 문장 속에 모
        토큰을 봄으로써 각 토큰의 의미를 정확히 모델을 전달하게 됩니다 즈 텍스트라는 단어와 메세지라는 단어와 한 께 있음으로 텍스트는 문자를 전송하다 라는 뜻이만 어텐션 백트를 통해서 전달하게 되고 역시 메세지라는 의미도 텍스트와 한 께 있는 것을 어텐션을 통해서 알게 되어가지구요 문자 메세지라는 의미로 다음 레이어로 전송하게 되죠 어텐션 백터는 플리커나 캐드 레이어로 전성이 되는데요 이와 통이란 거 
        여섯 번 진행돼요 그 최종 출력값은 트랜스포머 많이 디코드 입력값으로 사용해야 됩니다 여기서 꼭 기억해야 할 점 바로 인코더는 모든 토큰을 한 방에 계산한다는 점 왼쪽에서 오른쪽으로 하나씩 읽어가는 과정이 없다라는 점입니다 디코더는 인코드의 출력값과 최초 스타트 스페셜 토큰으로 작업을 시작합니다 디코더는 왼쪽부터 오른쪽으로 순차적으로 출략값을 생성합니다
        디코더는 이전 생성된 디코더의 출력 값과 인코더의 출력 값을 사용해서 현재의 출력 값을 생성해요 디코더 역시 어텐션 백도를 만들고요 클리콘네트들 레이어로 보내는 작업을 여섯 번 진행해요 디코더는 엔드 토큰이라는 스페셜 토큰을 출력할 때까지 그 안 보게 됩니다 자 이 과정을 통해서 imo 보이를 훌륭하게 한글로 가꾼 트랜스포머를 보고 계십니다 물론 언제가 트랜스포머를 너무 짧게 요약한 거 맞구요
        트랜스포머를 더 디테일하게 알고 싶은 분들은 제가 이전에 만든 영상 추천드릴게요 위에 링크 클릭하시면 트랜스포머 영상 보실 수 있습니다 이 정도까지만 트랜스포머 이해하셔도 벌트를 이해하신 데에는 충분할 걸 생각이 됩니다 다음 슬라이드로 넘어갈게요 자 트랜스포머에 인코더는 양방향으로 몸맥을 이해하고 디코더는 왼쪽에서 오른쪽으로 몸맥을 이해한다라는 게 핵심입니다 쉬운 질문 하나 드릴게요 벌트
        랭귀지 모델일까요 맞습니다 기존 많은 랭귀지 모델들이 왼쪽에서 오른쪽을 읽어 나가는 형태에 뭐 랭귀지 모델일 뿐 버튼은 양방향으로 학습 때는 기존 랭귀지 모델 거는 조금은 다른 형태이지만 랭귀지 모델이 맞습니다 기존 랭귀지 모델 제 담방향 랭귀지모델이 대표적인 이후로는 쥐 pt를 쓸 수 있겠죠 현재까지 읽은 단어들을 사용해서 다음 단어를 예측할 수 있도록 학습이 됩니다 입력된 문장을 통해서 입력값이 하와이일 경우에
        도잉을 예측카도로 학습이 됩니다 당면의 벌트는 동일한 문장 그대로 학습을 하되 가려진 단어를 예측카도로 학습이 됩니다 가려진 단어는 마스크 토큰이라고 불려요 벌트의 학습 역시 gp 티어와 마찬가지로 사람이 데이터를 레이블링할 필요가 없습니다 한순히 랜덤하게 문장 속 단어만 가려주고 가려진 단어를 맞추도록 학습하면 되는 것이죠 멀티에 있네 값으로는 한 문장뿐 아니라
        두 문장도 받을 수가 있습니다 한 문장 입력값을 받아 출력하는 대표적인 자연어처리 타스크로는 스팸인지 아닌지 문장이 근정적이지 부정적인지 불려하는 모델들이겠죠 드우 문장은 입력받는 대표적인 자연어처리 타스크로는 지금이 응답이 있습니다 입력값으로 질문과 정답이 들어있는 문맥을 받아서 정답을 출력하는 타스크지요 벌트는 한 문장 또는 두 문장에 학습데이터를 통해서 토큰 간에 상관 관계뿐 아니라 문장 간에 
        관관계도 학습하게 됩니다 여기 보이시는 cls는 클래스 티케이션 즉 블리오타스크에 사용되 위한 백터입니다 문장 전체가 하나의 벡터로 표현된 스페셜 토큰이다라고 이해하시면 되겠습니다 두 문장으로 구성돼 입력값의 경우에 sep라는 스페셜 토큰으로 두 문장이 구별됩니다 입력값을 조금 깊게 살펴보면 입력 토큰들이 포지션을 인코딩 그리고 세금을
        그 인베딩과 더해지는 것을 보실 수가 있어요 멀튼은 월드피스 인베딩 사용해서 문장을 토큰 단위로 분류해요 월드피스 인베딩은 단순히 띄어쓰기로 토큰을 나누는 것보다 효과적으로 토큰을 구분합니다 슬라이드에서 보시는 것처럼 플레잉은 하나의 단어이지만 플레이와 아이엔즈로 토큰이 나는 것을 볼 수가 있죠 이 같은 방법은 두 가지 장점이 있는데요 첫째에 플레이는 놀자라는 듯이
        고 아이엔지는 현재 무엇인가 하고 있다는 뜻이 명확하게 있기 때문에 딥랑인 모델에게 이 두 가지 의미는 명확히 전달할 수 있다는 장점이 있구요 둘째 이렇게 쪼개서 입력할 경우 신조어 또는 옷 탈자가 있는 입력값에도 예를 들어서 텍스팅 구글링처럼 사전에 없는 단어들도 텍스트와 i엔g 구글과 i엔지처럼 딥러인 모델이 학습당계에서 봤을 만한 단어들로 쪼개서 입력되기 때문에 흔치 않은 단어들에 대한 예측이 향상이 됩니다
        자 이번에는 세그멘트 인배딩을 알아볼까요 개념 아주 쉽습니다 두 개 문장이 입력될 경우에 각각의 문장에 서로 다른 숫자들을 더 해주는 게 바로 세그멘트 인배딩입니다 딥러니 모델에게 두 개 다른 문장이 있다는 것을 쉽게 알려주기 위해서 사용되는 인배딩입니다 포지션은 인배딩은 토큰들에 상대적 위치 정보를 알려줍니다 딥러니 모델은 포지션은 인배딩으로 이원 다음에 이투 이투 다음에 이쓰리 토큰이 위치함을 알
        이슈가 있습니다 포지셔널이 인배딩은 싸인 코싸인 함수를 사용하는데요 그게 세 가지 이유가 있어요 첫째 싸인과 꽃싸인의 출력 값은 임력 값에 사랑달라지죠 따라서 싸인 코싸인의 출력값은 김력 값에 상대적인 위치를 알 수 있는 숫자로 사용이 가능합니다 둘째 싸인과 코싸인의 출력 값은 규칙적으로 증가 또는 감소합니다 따라서 딥러니 모델이 이 규칙을 사용해서 입력값에 상대적 위
        쉽게 계산 가능하죠 셋째 싸인과 꽃스에 있는 무한대의 길이 입력값도 상대적인 위치를 출력할 수가 있어요 어떤 위치의 입력값이라도 마이너스 1에서 1 사이에 값을 출력하게 돼 있죠 포지션은 인배딩에 대해서 질문 중에 하나가 왜 상대적 위치를 사용하냐 절대적 위치를 사용하면 안 되냐라는 질문이 했는데요 예를 들어서 첫 번째 단어 일을 더라고 두 번째 단어의 이를 더라고 이런 식 말이죠 네 됩니다 그런데 왜 버튼 상대적 위치를 더 좋아할까요
        절대적 위치를 만약에 사용할 경우에는 최장길의 문장을 세팅해야 돼요 학습시 예상했던 최장길의 문장보다 더 큰 문장을 받을 수가 없게 되는 거죠 따라서 상대적 위치가 포지셔널 인 베딩에서 더 선호가 됩니다 벌튼은 프리트레이닝인 거야 파인트닝 이렇게 두 파트로 나입니다 그리고 방금 우리는 프리트레이닝 부분을 마스터 있죠 파인트닝을 알아보기 앞서서 간략하게 gpt와 벌트에 차이지 함 보고 갈게요 첫째 벌트는 양방향 랭귀지모 데리고 gp
        티는 단방향 랭귀지 모델입니다 둘째 벌트는 파인 투닝을 하기 위해서 만들어졌고 gpt는 파인 투닝이 필요 없도록 만들어졌습니다 그림으로 한번 이해해 볼까요 주황색 동그라미는 선행 학습된 즉 프리트레닝된 모델입니다 포시다시피 gpt는 선행 학습된 모델 그 자체로 여러 가지 목적의 자연어처리를 수행 가능합니다 다만 그 모델 크기가 상당히 크죠
        멀트는 모델이 상대적으로 잡구요 각각에 다른 자연어처리 위해서 따로 파인팅이 필요합니다 물론 문장을 단순히 분류하는 모델로 파인팅된 모델을 휴엔에 모델로도 사용할 수가 없죠 gpt는 한 번 학습시키는데 어마 한 시간과 돈이 됩니다 벌트는 상대적으로 적은 시간과 돈이 들지만 파인 투닝을 개발자가 해줘야 하고 파인 투닝이 역시 별도의 시간과 돈이 들어요 멀티에 파인 투닝을 알아볼까요
        선행학식 땐 벌트는 인터넷서 쉽게 다운로드 하실 수가 있습니다 개발자가 더 잘 알아야 하는 부분은 사실 이 다운로드 받은 모델을 어떻게 파인 투닝 할까입니다 벌튼 논문에 공개된 방법으로 몇 가지 파인 투닝 방법에 대해서 알아볼게요 첫 번째 예제는 두 문장에 관계를 예측하기 위해 모델을 만드는 방법입니다 벌트의 입력값으로 두 개의 문장을 받을 수 있는 걸 기억하시죠 두 개의 문
        sep 토큰으로 구분해서 벌팅 입력해서 출력값에 첫 번째 ces 토큰을 두 문장의 관계를 나타내도록 학습시킵니다 참 쉽죠 다음 예전에 이정 예전보다 식습니다 문장을 분류하는 모델 예제인데 문장 한계를 입력받고 ces 토큰이 분류값 중 하나가 되도록 학습시킵니다 이번 예전엔 큐앤에 적 지름이 응답 위재입니다 인력값으로 질문 거야 정답이 포함된 장문을
        sep 토큰으로 구분해서 줍니다 그리고 벌트의 출력값에 마지막 토큰들이 장문 속에 위치한 정답에 시작 인텍스와 마지막 인텍스를 출력 카드로 학습을 시키니다 마지막 예제는 문장 속 단어를 태닝하는 예재입니다 각각의 입력 토큰에 대한 출력값이 있기 때문에 이 출력값이 원하는 태킹으로 출력 되도록 학습을 시키니다 벌트의 성능은 상당히 우수합니다 gpt 원과 동일한 사이즈의 벌트가 gpt 원보다
        높은 성능을 가졌구요 심지어 더 큰 모델은 더 높은 성능을 보여주네요 끝까지 봐주셔서 감사니다 모든 내용은 어트에 논문을 참조해서 만들었구요 다음 비도에서 조금 더 재밌는 토페이으로 찾아올게요 감사니다"""

        inputs = [args.prefix + sample]
        inputs = tokenizer(inputs, max_length=args.max_input_length, truncation=True, return_tensors="pt")
        output = model.generate(**inputs, num_beams=8, do_sample=True, min_length=10, max_length=args.max_target_length)
        decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        result = nltk.sent_tokenize(decoded_output.strip())[0]
        print('First Example Result:')
        print(result)


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
        print('Second Example Result:')
        print(result)