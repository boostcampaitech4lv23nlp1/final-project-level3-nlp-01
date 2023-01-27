# 요약 후처리
# input : 요약 문장 묶음 -> output : 후처리된 요약 문단
# TODO: 반복문 제거
# TODO: koelectra (masking) + kobart (masking filling)
from tqdm import tqdm
from itertools import combinations
import torch
from transformers import ElectraForPreTraining, ElectraTokenizer
from transformers import pipeline
from .preprocess import PreProcessor


class PostProcessor(PreProcessor):

    def delete_loop(self, summary):
        new_summary = summary[:]
        tmp = summary.split()
        arr = [i for i in range(len(tmp)+1)]
        can_list = [com for com in combinations(arr, 2) if com[1] - com[0] + 1 <= len(arr)-com[1]]
        for can in can_list:
            string = tmp[can[0]:can[1]]
            stick = can[1]
            len_string = len(string)
            cnt = 0
            for i in range((len(arr) - can[1]) // (can[1] - can[0])):
                end_stick = stick + len_string
                if string != tmp[stick:end_stick]:
                    continue
                cnt += 1
                stick = end_stick
            if cnt != 0:
                new_tmp = tmp[:can[1]] + tmp[can[1] + len_string*cnt:]
                new_summary = ' '.join(new_tmp)
                break
        if summary == new_summary:
            return summary
        else:
            return self.delete_loop(new_summary)
    
    def awkward_subs(self, summary, threshold=0.6):
        discriminator = ElectraForPreTraining.from_pretrained("monologg/koelectra-small-v3-discriminator")
        disc_tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-small-v3-discriminator")
        substitutor = pipeline("fill-mask",
                                    model = "monologg/koelectra-small-v3-generator",
                                    tokenizer = ("monologg/koelectra-small-v3-generator", {'use_fast':True}),
                                    framework ='pt', # pytorch tensor 형태 반환
                                    top_k = 1)
        tokenized_summary = disc_tokenizer.tokenize(summary)
        encoded_summary = disc_tokenizer.encode(summary, return_tensors = 'pt') # pytorch tensor 형태 반환

        disc_summary = discriminator(encoded_summary)
        sigmoid_summary = torch.sigmoid(disc_summary[0])
        masked_tokens = ["[MASK]" if x>=threshold else tokenized_summary[i] for i, x in enumerate(sigmoid_summary[0][1:-1])]
        masked_summary = disc_tokenizer.convert_tokens_to_string(masked_tokens)

        if "[MASK]" in masked_tokens: # 어색한 표현이 [MASK] 토큰으로 변경된 경우 -> 변경한 summary 반환
            subs = substitutor(masked_summary)
            subs_cnt = len(subs)

            if subs_cnt == 1:
                subs_summary = subs[0]['sequence']
            elif subs_cnt > 1:
                for i in range(subs_cnt):
                    replaced = subs[i][0]['token_str']
                    masked_summary = masked_summary.replace("[MASK]", replaced, 1)
                subs_summary = masked_summary
            subs_summary = subs_summary.replace(' .', '.')
            subs_summary = subs_summary.replace(' ##', '')
            return subs_summary

        return summary # 어색한 표현이 없는 경우 -> 원본 summary 반환
     
    def postprocess(self):
        all_subs = []
        for data in tqdm(self.data, desc='postprocessing'):
            del_loop_data = self.delete_loop(data)
            all_subs.append(del_loop_data)
        phrases = self.phrase_split(all_subs)
        return phrases

        



if __name__ == '__main__':
    import pickle
    with open('/opt/ml/Summarization/sum.pickle', 'rb') as f:
        data = pickle.load(f)
    postprocesser = PostProcessor(data = data, stride = 4, min_len=2, max_len=512)
    postprocesser.postprocess()