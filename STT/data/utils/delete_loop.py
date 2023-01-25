from itertools import combinations

def delete_loop(text):
    new_text = text[:]
    tmp = text.split()
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
            new_text = ' '.join(new_tmp)
            break
    if text == new_text:
        return text
    else:
        return delete_loop(new_text)