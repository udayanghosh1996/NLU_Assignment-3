import datasets
from transformers import AutoTokenizer
import nltk
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
from pycocoevalcap.cider.cider import Cider

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
rouge = datasets.load_metric("rouge")

encoder_max_length=512
decoder_max_length=128
def process_data_to_model_inputs_summarization(batch):
  # tokenize the inputs and labels
  inputs = tokenizer(batch["input"], padding="max_length", truncation=True, max_length=encoder_max_length)
  outputs = tokenizer(batch["target"], padding="max_length", truncation=True, max_length=decoder_max_length)
  batch["input_ids"] = inputs.input_ids
  batch["attention_mask"] = inputs.attention_mask
  batch["decoder_input_ids"] = outputs.input_ids
  batch["decoder_attention_mask"] = outputs.attention_mask
  batch["labels"] = outputs.input_ids.copy()
  # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`.
  # We have to make sure that the PAD token is ignored
  batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]
  return batch
  
def process_data_to_model_inputs_translation(batch):
  inputs = ["translate English to Hindi: " + ex["en"] for ex in batch["translation"]]
  targets = [ex["hi"] for ex in batch["translation"]]
  # tokenize the inputs and labels
  inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=encoder_max_length)
  outputs = tokenizer(targets, padding="max_length", truncation=True, max_length=decoder_max_length)
  batch["input_ids"] = inputs.input_ids
  batch["attention_mask"] = inputs.attention_mask
  batch["decoder_input_ids"] = outputs.input_ids
  batch["decoder_attention_mask"] = outputs.attention_mask
  batch["labels"] = outputs.input_ids.copy()
  # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`.
  # We have to make sure that the PAD token is ignored
  batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]
  return batch


def compute_metrics(pred):
  labels_ids = pred.label_ids
  pred_ids = pred.predictions
  pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
  labels_ids[labels_ids == -100] = tokenizer.pad_token_id
  label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
  rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid
  return {"rouge2_precision": round(rouge_output.precision, 4),
      "rouge2_recall": round(rouge_output.recall, 4),
      "rouge2_fmeasure": round(rouge_output.fmeasure, 4)}

def calculate_cider_scores(actual, generated):
  act_dict = {idx: [line] for idx, line in enumerate(actual)}
  gen_dict = {idx: [line] for idx, line in enumerate(generated)}
  cider = Cider()
  (score, scores) = cider.compute_score(act_dict, gen_dict)
  return score

def calculate_bleu_scores(actual, generated):
  smoothie = SmoothingFunction().method4
  actual_tokenized = [[nltk.word_tokenize(act) for act in group] for group in actual]
  generated_tokenized = [nltk.word_tokenize(gen) for gen in generated]
  score = corpus_bleu(actual_tokenized, generated_tokenized, smoothing_function=smoothie)
  return score
