import datasets
from datasets import load_dataset

def load_data(base_path):
    dataset = load_dataset("json", data_files={'train':base_path+'train.json', 'validation':base_path+'val.json', 'test':base_path+'test.json'})

    n_samples_train = len(dataset["train"])
    n_samples_validation = len(dataset["validation"])
    n_samples_test = len(dataset["test"])
    n_samples_total = n_samples_train + n_samples_validation + n_samples_test

    print(f"- Training set:\t{n_samples_train} ({n_samples_train*100/n_samples_total:.2f}%)")
    print(f"- Val set:\t{n_samples_validation} ({n_samples_validation*100/n_samples_total:.2f}%)")
    print(f"- Test set:\t{n_samples_test} ({n_samples_test*100/n_samples_total:.2f}%)")

    return dataset
    