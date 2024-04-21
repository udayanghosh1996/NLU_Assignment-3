from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
import datasets
from get_dataset import *
from utils import *
from model import *

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

training_args = Seq2SeqTrainingArguments( predict_with_generate=True,
                                          evaluation_strategy="steps",
                                          per_device_train_batch_size=batch_size,
                                          per_device_eval_batch_size=batch_size,
                                          fp16=True, #fro GPU, False for CPU
                                          output_dir="./",
                                          logging_steps=2,
                                          save_steps=10,
                                          eval_steps=4)

rouge = datasets.load_metric("rouge")
# instantiate trainer
trainer = Seq2SeqTrainer(model=bert2bert,
                        args=training_args,
                        compute_metrics=compute_metrics,
                        train_dataset=tokenized_datasets['train'],
                        eval_dataset=tokenized_datasets['valid'])

#training summarization model
trainer.train()
#save model checkpoint
bert2bert.save_pretrained("bert2bert_summarization")

#evaluation
def generate_summary(batch):
  inputs = tokenizer(batch["input"], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
  input_ids = inputs.input_ids.to("cuda")
  attention_mask = inputs.attention_mask.to("cuda")
  outputs = bert2bert.generate(input_ids, attention_mask=attention_mask)
  output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
  batch["pred_summary"] = output_str
  return batch

batch_size = 16 # change to 64 for full evaluation
results = tokenized_datasets['test'].map(generate_summary, batched=True, batch_size=batch_size, remove_columns=["input"])

#get rouge scores
print("Rouge Scores : \n", rouge.compute(predictions=results["pred_summary"],
                                         references=results["target"], rouge_types=["rouge2"])["rouge2"].mid)

#get cider scores
cider = calculate_cider_scores(results["target"], results["pred_summary"])
print("Cider Scores : \n", cider)

#get bleu scores
bleu = calculate_bleu_scores(results["target"], results["pred_summary"])
print("Bleu Scores : \n", bleu)
