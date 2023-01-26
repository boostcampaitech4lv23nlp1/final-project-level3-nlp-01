import pandas as pd


def custom_split(dfs: pd.DataFrame) -> pd.DataFrame:
    text = " ".join([item['output'] for _, item in dfs.iterrows()])
    
    outputs, output = [], ""
    end_chars = ['.', '?']
    end_flag = False
    max_length = 20
    for c in text:
        if end_flag is False:
            if c == ' ' and len(output) > max_length:
                try:
                    if len(outputs[-1]) < max_length:
                        outputs[-1] += output
                    else:
                        outputs.append(output + ' ')
                except IndexError as indexError:
                    outputs.append(output + ' ')
                output = ""
                continue
            if c in end_chars:
                end_flag = True
            output += c
        else:
            if c in end_chars:
                output += c
            else:
                if len(outputs) < 1:
                    outputs.append(output + ' ')
                else:
                    if len(outputs[-1]) < max_length:
                        outputs[-1] += output
                    else:
                        outputs.append(output + ' ')
                output = ""
                end_flag = False
    dfs = pd.DataFrame({
        'label': ['' for _ in range(len(outputs))],
        'output': outputs
    })
    return dfs