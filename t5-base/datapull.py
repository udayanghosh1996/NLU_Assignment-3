# datapull.py

import pandas as pd
from datasets import Dataset , DatasetDict
from datasets import load_dataset

def pull_data(english_file_path, hindi_file_path,task = "translation"):
    
    if task == "translation" : 
        print("Loading English-Hindi Translation Data : ")
        # Read English data from file
        with open(english_file_path, "r", encoding="utf-8") as f:
            english_data = f.readlines()[:15000]

        # Read Hindi data from file
        with open(hindi_file_path, "r", encoding="utf-8") as f:
            hindi_data = f.readlines()[:15000]

        # Create DataFrame
        data = {"translation": [{'en' : text_en , 'hi' : text_hi} for (text_en,text_hi) in zip(english_data,hindi_data)]}
        df = pd.DataFrame(data)

        # Create Hugging Face Dataset
        dataset = Dataset.from_pandas(df)
    
    elif task == "summarization" : 
        print("Loading IndicHeadlineGeneration Data : ")
        
        dataset = load_dataset("ai4bharat/IndicHeadlineGeneration",'hi',split='train[:15000]')
        
    
    return dataset
