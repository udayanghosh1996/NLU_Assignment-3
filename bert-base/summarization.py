from get_dataset import *
from utils import *

#taking first 10,000 datapoints for training.
sample_size = 10000
hindi_file_path = None
english_file_path  = None
task = 'summarization'
sum_dataset = get_data(hindi_file_path , english_file_path, task, sample_size)

## Train-Test-Val Split  for summarization
train_test_dataset = sum_dataset.train_test_split(test_size=0.15)
test_valid = train_test_dataset['test'].train_test_split(test_size=0.5)
sum_raw_datasets = DatasetDict({'train': train_test_dataset['train'],
                            'test': test_valid['test'],
                            'valid': test_valid['train']})

# Tokenize datasets
batch_size=16
tokenized_datasets = sum_raw_datasets.map(process_data_to_model_inputs_summarization, batched=True,
                                            batch_size=batch_size,remove_columns=["input", "target"])
tokenized_datasets.set_format(type="torch",
                          columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"])

print("final datasets (train, test, valid sets) : ", tokenized_datasets)

