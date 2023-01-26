import pandas as pd
import math
import os

def split_file(dfs,split):
    
    div = math.ceil(len(dfs) / split)
    left, right = 0, div
    idx = 0
    scps = []
    if os.path.exists('./output'):
        pass
    else:
        os.makedirs('./output')
    while True:
        scp = os.path.join('./output',f'data_{idx}.csv')
        if idx == split-1 :
            df = dfs[left:]
        else:
            df = dfs[left:right]
        
        if len(dfs[left:right]) > 0:
            scps.append(scp)
            df.to_csv(scp,index=False)
        
        if right < 0 or split == 1: break
        
        left = right
        
        if right + div < len(dfs):
            right += div
        else:
            right = -1
            
        idx += 1
        
    return scps
          

if __name__ == '__main__' : 
    
    scps = split_file(dfs = pd.read_csv('history_dataset.csv'),split=8)
    print(scps)