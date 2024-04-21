import pandas as pd
from datasets import Dataset , DatasetDict,load_dataset
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-hi")

# map input and target len to dict as well as get size of samples.
def get_length(x):
  x["article_len"] = len(tokenizer(x["input"]).input_ids)
  x["article_longer_512"] = int(x["article_len"] > tokenizer.model_max_length)
  x["summary_len"] = len(tokenizer(x["target"]).input_ids)
  x["summary_longer_64"] = int(x["summary_len"] > 64)
  x["summary_longer_128"] = int(x["summary_len"] > 128)
  return x

def compute_stats(x,sample_size = 1000):
  if len(x["article_len"]) == sample_size:
    print("Article Mean:",sum(x["article_len"]) / sample_size)
    print(" %-Articles > 512:{}",sum(x["article_longer_512"]) / sample_size)
    print("Summary Mean:{}",sum(x["summary_len"]) / sample_size)
    print(" %-Summary > 64:{}",sum(x["summary_longer_64"]) / sample_size)
    print("%-Summary > 128:{}",sum(x["summary_longer_128"]) / sample_size)

def get_data(hindi_file_path , english_file_path, task, size):
  if task == "summarization" :
    #get indicheadline generation dataset for hindi
    dataset = load_dataset("ai4bharat/IndicHeadlineGeneration",'hi',split=f'train[:{size}]')
    #compute and print lengths of tokens in dataset for EDA
    data_stats = dataset.map(get_length, num_proc=4)
    data_stats.map(compute_stats, batched=True,batch_size=-1)

  elif task == "translation" :
    # Read English data (subsample only using size variable ) from file
    with open(english_file_path, "r", encoding="utf-8") as f:
        english_data = f.readlines()[:size]
    # Read Hindi data (subsample only using size variable ) from file
    with open(hindi_file_path, "r", encoding="utf-8") as f:
        hindi_data = f.readlines()[:size]
    # Create DataFrame
    data = {"translation": [{'en' : data_en , 'hi' : data_hi} for (data_en,data_hi) in zip(english_data,hindi_data)]}
    df = pd.DataFrame(data)
    # Create Hugging Face Dataset
    dataset = Dataset.from_pandas(df)
  return dataset
