import numpy as np
from datasets import load_metric

prefix = "translate English to Hindi: "
max_input_length = 512
max_target_length = 512
source_lang = "en"
target_lang = "hi"
metric_r = load_metric("rouge")
metric_b = load_metric("sacrebleu")



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
    result = {"rouge1": result["rouge1"].mid.fmeasure}
    
    bleu = metric_b.compute(predictions=decoded_preds, references=decoded_labels)
    result['bleu'] = bleu['score']
    
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result