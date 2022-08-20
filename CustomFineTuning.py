"""
TFE - Chatbot Tifi - Technifutur
by Nicolas Christiaens
"""

from datasets import load_dataset
from sentence_transformers import SentenceTransformer, models
from sentence_transformers.readers import InputExample
from torch.utils.data import DataLoader
from typing import Iterable, Dict
from torch import nn, Tensor
import torch
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import math
from FineTuning import BiEncoder
import pandas as pd
from Preprocessing import Preprocessing

# Function to read the custom dataset and transform it in a dataframe
def readCustomDS(file):
    df = pd.read_excel(file)
    return df

# Function to process the custom dataset to have good input formats
def processCustomDS(df):
    process_dataset = []
    
    for index,row in df.iterrows():
        score = float(row.Score)
        sample = InputExample(texts=[Preprocessing(row.Sentence1),Preprocessing(row.Sentence2)],label=score)
        process_dataset.append(sample)
    
    return process_dataset
    

if __name__ == "__main__":
    # Inform the user if no GPU is detected
    if torch.cuda.is_available() is True:
        device = "cuda"
    else:
        print("Pas de GPU pour le training")
        
    # Read the custom dataset
    file = "customDS.xlsx"
    df = readCustomDS(file)
    
    # Get the custom dataset
    train_ds = processCustomDS(df)

    # Use DataLoader for training
    train_batch_size = 1
    trainloader = DataLoader(train_ds,
                             shuffle=True,
                             batch_size=train_batch_size,
                             num_workers=0)
    
    # Use our model checkpoint
    model_name =  "Model_Finetuning/camembert-base"
    model_save = "Model_Custom/camembert-base"
    max_seq_length = 128
    word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
    
    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=False,
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_mean_sqrt_len_tokens=False)
    model = SentenceTransformer(modules=[word_embedding_model,pooling_model])
    
    # Define epochs and train loss
    num_epochs = 10
    train_loss = BiEncoder(model=model)
    
    # Train the model
    model.fit(train_objectives=[(trainloader,train_loss)],
              epochs=num_epochs,
              evaluation_steps=1000,
              optimizer_params={'lr': 1e-5, 
                                'eps': 1e-6, 
                                'correct_bias': False},
              save_best_model = True,
              output_path=model_save)