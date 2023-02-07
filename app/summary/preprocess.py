import kss
from sentence_transformers import util

class PreProcessor:
    def __init__(self, model, data, stride=4, min_len=300, max_len=768):
        self.model = model
        self.data = data
        self.stride = stride
        self.min_len = min_len
        self.max_len = max_len

    def sentence_split(self):
        sent_data = kss.split_sentences(self.data, backend = 'mecab', num_workers=8)
        return sent_data # list 형태
        
    def phrase_split(self, sent_data):

        phrase_list = []
        for i in range(len(sent_data)):
            tmp = [x.strip() for x in sent_data[i:i+self.stride]]
            tmp_phrase = ' '.join(tmp)
            phrase_list.append(tmp_phrase)
        model = self.model
        vectors = model.encode(phrase_list)

        similarities = util.cos_sim(vectors, vectors)
        sim_list = []
        for i in range(similarities.size()[0]-self.stride):
            sim_list.append([float(similarities[i][i+1]), i])
        s_similarities = sorted(sim_list, key=lambda x : x[0])

        sp = [0, len(phrase_list)] # split_point
        done_split = []
        finished = []
        record = []

        def make_split():
            nonlocal finished
            nonlocal sp
            nonlocal done_split
            nonlocal record

            _, point = s_similarities.pop(0) # 가장 유사도 낮은 지점
            point += self.stride # 나눌 위치 추가
            if point in done_split: # 나눠놓은 문장을 나누려고 하면 종료
                return

            sp_copy = sp[:] # sp_copy
            sp_copy.append(point) # 분리할 point append
            sp_copy.sort() # point 정렬
            stick = sp_copy.index(point) # 기준이 되는 인덱스 가져옴
            up_p = sp_copy[stick-1] # 기준점 위쪽의 문서들 시작 위치
            down_p = sp_copy[stick+1] # 기준점 아래쪽의 문서들 끝 위치
            up = sent_data[up_p:point]
            down = sent_data[point: down_p]
            up_len = sum([len(u_t) for u_t in up])
            down_len = sum([len(d_t) for d_t in down])
            if up_len < self.min_len or down_len < self.min_len: # min_len 보다 작으면 자르지 않음
                return
            
            sp = sp_copy # point가 자를 수 있으니 sp_copy를 sp로 할당
            if up_len <= self.max_len: # min ~ max 이면 기록
                done_split += [p for p in range(up_p,point)]
                record.append([up_p, point, up_len])
                finished.append([sent_data[up_p:point], up_p])
                record.sort(key = lambda x : x[0])

            if down_len <= self.max_len: # min ~ max 이면 기록
                done_split += [p for p in range(point,down_p)]
                record.append([point, down_p, down_len])
                finished.append([sent_data[point:down_p], point])
                record.sort(key = lambda x : x[0])

        all_done = [i for i in range(len(phrase_list))]
        while True:

            done_split = sorted(done_split)
            if done_split == all_done:
                break
            make_split()

        finished.sort(key=lambda x : x[1])
        finished = [' '.join(x[0]) if len(' '.join(x[0])) > self.min_len else None for x in finished] # 문단의 길이가 min_len 이하면 제거하기 [동전 10000개 ...]
        try:
            finished.remove(None)
        except:
            pass
        return finished

    def preprocess(self):
        sent_data = self.sentence_split()
        phrase_data = self.phrase_split(sent_data)

        return phrase_data


if __name__ == '__main__':
    import pandas as pd
    from sentence_transformers import SentenceTransformer
    data = pd.read_csv('/opt/ml/level3_productserving-level3-nlp-01/output/inference_large_beam10_len20.csv', index_col = 0)
    
    data['result'] = data['result'].astype(str)
    data = ' '.join(data['result'])
    model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
    preprocesser = PreProcessor(model = model, data = data, stride = 4, min_len=300, max_len=1000)
    split_data = preprocesser.preprocess()
    # print(split_data)
    print(len(split_data))
    # with open('/opt/ml/output/seg_300.pickle', 'wb') as f:
    #     pickle.dump(split_data, f)
    
