from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
import datasets
from get_dataset import *
from utils import *
from model import *

#taking first 10,000 datapoints for training.
sample_size = 10000
hindi_file_path = '/content/drive/MyDrive/train.en'
english_file_path  = '/content/drive/MyDrive/train.hi'
task = 'translation'
trans_dataset = get_data(hindi_file_path , english_file_path, task, sample_size)

# Train-Test-Val Split  for translation
train_test_dataset = trans_dataset.train_test_split(test_size=0.15)
test_valid = train_test_dataset['test'].train_test_split(test_size=0.5)
trans_raw_datasets = DatasetDict({'train': train_test_dataset['train'],
                            'test': test_valid['test'],
                            'valid': test_valid['train']})

# Tokenize datasets
batch_size=16
tokenized_datasets = trans_raw_datasets.map(process_data_to_model_inputs_translation, batched=True)
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
bert2bert.save_pretrained("bert2bert_translation")

#evaluation
def generate_summary(batch):
  inputs = ["translate English to Hindi: " + ex["en"] for ex in batch["translation"]]
  targets = [ex["hi"] for ex in batch["translation"]]
  inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
  input_ids = inputs.input_ids.to("cuda")
  attention_mask = inputs.attention_mask.to("cuda")
  outputs = bert2bert.generate(input_ids, attention_mask=attention_mask)
  output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
  batch["translated_txt"] = output_str
  batch["target_txt"] = targets
  return batch

batch_size = 16 # change to 64 for full evaluation
results = tokenized_datasets['test'].map(generate_summary, batched=True, batch_size=batch_size)

#get rouge scores
print("Rouge Scores : \n", rouge.compute(predictions=results["translated_txt"],
                                         references=results["target_txt"], rouge_types=["rouge2"])["rouge2"].mid)

#get cider scores
cider = calculate_cider_scores(results["target_txt"], results["translated_txt"])
print("Cider Scores : \n", cider)

#get bleu scores
bleu = calculate_bleu_scores(results["target_txt"], results["translated_txt"])
print("Bleu Scores : \n", bleu)
