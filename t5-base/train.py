# train.py

import pandas as pd
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer


def fine_tune_model(tokenized_datasets, args, data_collator,tokenizer,model,compute_metrics,model_path,task):

    # Create Seq2SeqTrainer
    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Train the model
    output = trainer.train()
    
    df_logs = pd.DataFrame(trainer.state.log_history)
    
    df_logs.to_excel(f"./{task}_training_logs.xlsx")

    # Save the trained model
    trainer.save_model(model_path)
