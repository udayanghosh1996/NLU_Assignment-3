# eval.py

from transformers import AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, AutoTokenizer

def evaluate_model(model_path, input_text,ground_truth,task):
    print(f"{task.upper() } Evaluation : ")
    
    # Load the model checkpoint
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    # Define the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Tokenize the input text
    input_tokens = tokenizer.batch_encode_plus(input_text, max_length=512, truncation=True, return_tensors="pt",padding=True)

    # Perform inference
    outputs = model.generate(input_ids=input_tokens["input_ids"], attention_mask=input_tokens["attention_mask"],max_length= 128,early_stopping = True)

    # Decode the generated output tokens
    output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    for inp,out,gt in zip(input_text,output_text,ground_truth) :
        print('*'*100)
        print()
        print(f"{task.upper() } Model input : {inp} ")
        print(f"{task.upper() } Model output : {out} ")
        print(f"Ground Truth : {gt} ")
        print()
        print('*'*100)
