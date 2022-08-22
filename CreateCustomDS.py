"""
TFE - Chatbot Tifi - Technifutur
by Nicolas Christiaens
"""

from FineTuning import STSBTrainingModel
from transformers import AutoModel,AutoTokenizer
import torch
import pandas as pd
from openpyxl import load_workbook
from Preprocessing import Preprocessing

# Function to create a file for the dataset if it doesn't exist
def createTrainSet(file):
    try:
        open(file)
    except:
        df = pd.DataFrame(columns=["sentence1","sentence2","score","Old Similarity"])
        df.to_excel(file,index=False)

# Function to save a sample in the dataset
def saveSample(file,sentence1,sentence2,score,sim):
    # Get the log
    new_sample = [sentence1,sentence2,score,sim]
    
    # Save the log
    wb = load_workbook(file)
    page = wb.active
    page.append(new_sample)
    wb.save(filename=file)

if __name__ == "__main__":
    # Load model
    model_name = "Model_SentenceEmbedding/Finetuning/Final_model"
    max_length = 128
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create the tokenize function
    def tokenize(text):
        return tokenizer([text],padding=True,truncation=True,max_length=max_length,return_tensors="pt")

    # Load the model used as body
    body = AutoModel.from_pretrained(model_name,max_length=max_length)

    # Create the training model
    model = STSBTrainingModel(body=body)
    
    # Make prediction
    Savefile = "customDS.xlsx"
    createTrainSet(Savefile)
    stop = True
    print("Nous allons commencer le test de similarité entre des phrases spécifiques et les ajouter à la base de donnée si nécessaire")
    while stop:
        # Take inputs
        sentence1 = input("Entrez la 1er phrase\n")
        sentence2 = input("Entrez la 2eme phrase\n")
        
        # Make the Preprocessing
        sentence1 = Preprocessing(sentence1)
        sentence2 = Preprocessing(sentence2)
        
        # Tokenize
        token1 = tokenize(sentence1)
        token2 = tokenize(sentence2)
        
        # Create the sample batch
        batch = {}
        batch["input_ids1"] = token1["input_ids"]
        batch["attention_mask1"] = token1["attention_mask"]
        batch["input_ids2"] = token2["input_ids"]
        batch["attention_mask2"] = token2["attention_mask"]
        batch["score"] = None
        
        # Compute the cosinus similarity
        with torch.no_grad():
            _,sim = model(**batch)
        sim = sim[0]
        sim = sim.numpy()
        print(sim)
        print("Similarité : " + str(sim))
        
        # Ask user to record the inputs in the dataset or not
        record = input("Enregistrer o ou n\n")
        if record == "o":
            scoreIsOk = True
            while scoreIsOk:
                try:
                    # Ask user the new score of the paired inputs
                    score = input("Donner un score de similarité entre 0 (minimum) et 1 (maximum)\n")
                    score = float(score)
                    scoreIsOk = False
                except:
                    print("Donner un nombre valide")
            # Change the score in case of bad use
            if score < 0:
                score = 0
            if score > 1:
                score = 1
            # Save in the files
            saveSample(Savefile,sentence1,sentence2,str(score),str(sim))
        
        # Ask user if he wants to continue or not
        stop_in = input("Voulez vous stopper ? o ou n\n")
        if stop_in == "o":
            stop = False