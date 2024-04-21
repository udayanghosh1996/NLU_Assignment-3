from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-hi")

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
