from transformers import EncoderDecoderModel
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
bert2bert = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-uncased", "bert-base-uncased")

# Freeze all model weights except for the word embedding layer
for param in bert2bert.parameters():
    param.requires_grad = True
bert2bert.get_encoder().requires_grad = True

# Resize token embeddings for both encoder and decoder
bert2bert.get_encoder().resize_token_embeddings(len(tokenizer))
bert2bert.get_decoder().resize_token_embeddings(len(tokenizer))

print("Number of  Params in bert2bert model: ", bert2bert.num_parameters())

bert2bert.config.decoder_start_token_id = tokenizer.cls_token_id
bert2bert.config.eos_token_id = tokenizer.sep_token_id
bert2bert.config.pad_token_id = tokenizer.pad_token_id
bert2bert.config.vocab_size = bert2bert.config.encoder.vocab_size

bert2bert.config.max_length = 142
bert2bert.config.min_length = 56
bert2bert.config.no_repeat_ngram_size = 3
bert2bert.config.early_stopping = True
bert2bert.config.length_penalty = 2.0
bert2bert.config.num_beams = 4

