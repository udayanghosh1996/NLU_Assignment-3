# main.py

from datapull import pull_data
from train import fine_tune_model
from evaluate import evaluate_model
from transformers import Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, AutoTokenizer
from datasets import Dataset , DatasetDict
from transformers import T5ForConditionalGeneration
import numpy as np
from datasets import load_metric
import warnings
warnings.filterwarnings("ignore")
import argparse

# Create the argparse object
parser = argparse.ArgumentParser(description='Argument Parser')

# Add the arguments
parser.add_argument('--task', type=str, choices=['translation', 'summarization'], help='The task to perform')
parser.add_argument('--epoch', type=int, default=10, help='Number of epochs')
parser.add_argument('--datasize', type=int, default=15000, help='Size of the data')

# Parse the arguments
args = parser.parse_args()

# Access the arguments
batch_size = 2
task = args.task
epochs = args.epoch
datasize = args.datasize

print(f'Task: {task}')
print(f'Epoch: {epochs}')
print(f'Datasize: {datasize}')

ft_model_name = 't5-base'
model_checkpoint = "Helsinki-NLP/opus-mt-en-hi"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
metric_r = load_metric("rouge")
metric_b = load_metric("sacrebleu")

if task == "translation" : 
    prefix = "translate English to Hindi: "
    max_input_length = 400
    max_target_length = 400
    source_lang = "en"
    target_lang = "hi"
    
elif task == "summarization" :
    prefix = "summarize: "
    max_input_length = 400
    max_target_length = 128


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric_r.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"rouge1": result["rouge1"].mid.fmeasure, "rouge2": result["rouge2"].mid.fmeasure, "rougeL": result["rougeL"].mid.fmeasure}
    
    bleu = metric_b.compute(predictions=decoded_preds, references=decoded_labels)
    result['bleu'] = bleu['score']
    
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

def preprocess_function_summarize(examples):
    inputs = [doc for doc in examples["input"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    # labels = tokenizer(text_target=examples["target"], max_length=max_target_length, truncation=True)
    
    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(text_target=examples["target"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return  model_inputs

def preprocess_function_translation(examples):
    inputs = [prefix + ex[source_lang] for ex in examples["translation"]]
    targets = [ex[target_lang] for ex in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def data_paths(task) :
    # Define paths and model names
    if task == "translation" :
        finetune_model = "en-hi-translation"
        model_path = "./translation_model"
        english_file_path = "./train.en"
        hindi_file_path = "./train.hi"
        model_save_path = "/home/jupyter/duplicates_detection/intl-duplicates/det_lat/t5_translation.pkl"
        
    elif task == "summarization" :
        finetune_model = "headline-summarize"
        model_path = "./summarize_model"
        model_save_path = "/home/jupyter/duplicates_detection/intl-duplicates/det_lat/t5_summarize.pkl"
        english_file_path = None
        hindi_file_path = None
    
    return finetune_model , model_path , english_file_path , hindi_file_path , model_save_path
    

def main(task = "translation"):
    
    ## Defining Paths
    finetune_model , model_path , english_file_path , hindi_file_path , model_save_path = data_paths(task)
    # Pull data
    print("Fetching Data : ")
    dataset = pull_data(english_file_path, hindi_file_path,task)

    ## Train-Test-Valiation Split 
    print("Train-Test-Valiation Split : ")
    train_test_dataset = dataset.train_test_split(test_size=0.15)
    test_valid = train_test_dataset['test'].train_test_split(test_size=0.5)
    raw_datasets = DatasetDict({'train': train_test_dataset['train'],
                                'test': test_valid['test'],
                                'valid': test_valid['train']})
    # Tokenize datasets
    print("Tokenization : ")

    if task == "translation" : 
        tokenized_datasets = raw_datasets.map(preprocess_function_translation, batched=True)
    elif task == "summarization" : 
        tokenized_datasets = raw_datasets.map(preprocess_function_summarize, batched=True)


    # Define training arguments
    args = Seq2SeqTrainingArguments(
        f"{task}-{finetune_model}-finetuned-15apr",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-3,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.05,
        save_total_limit=2,
        num_train_epochs=epochs,
        predict_with_generate=True,
        fp16=True,
        push_to_hub=False,
        generation_max_length=128,
        load_best_model_at_end=True
    )
    
    # Define model
    model = T5ForConditionalGeneration.from_pretrained(ft_model_name)

    # Freeze all model weights except for the word embedding layer
    for param in model.parameters():
        param.requires_grad = True
    model.get_encoder().embed_tokens.weight.requires_grad = True
    model.resize_token_embeddings(len(tokenizer))

    # Create data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Fine-tune the model
    print("Finetuning : ")
    fine_tune_model(tokenized_datasets, args, data_collator,tokenizer,model,compute_metrics,model_path,task)

    # Evaluate the model
    if task == "translation" : 
        input_text_list = [list(hi.values())[0] for hi in  raw_datasets['test']['translation']][:10]
        gt_list = [list(hi.values())[1] for hi in  raw_datasets['test']['translation']][:10]
    elif task == "summarization" : 
        input_text_list = [raw_datasets['test'][i]['input'] for i in range(10)]
        gt_list = [raw_datasets['test'][i]['target'] for i in range(10)]
        
    output_text = evaluate_model(model_path, input_text_list,gt_list,task)
    print(output_text)

if __name__ == "__main__":
    main(task)
