"""
TFE - Chatbot Tifi - Technifutur
by Nicolas Christiaens
"""

from sentence_transformers import SentenceTransformer, models
import torch
import pandas as pd
from openpyxl import load_workbook
from sklearn.metrics.pairwise import cosine_similarity
from Preprocessing import Preprocessing

# Function to create a file for the dataset if it doesn't exist
def createTrainSet(file):
    try:
        open(file)
    except:
        df = pd.DataFrame(columns=["Sentence1","Sentence2","Score","Old Similarity"])
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
    model_name = "Model_Custom/camembert-base"
    model = SentenceTransformer(model_name)
    
    # Make prediction
    file = "customDS.xlsx"
    createTrainSet(file)
    stop = True
    print("Nous allons commencer le test de similarité entre des phrases spécifiques et les ajouter à la base de donnée si nécessaire")
    while stop:
        # Take inputs
        sentence1 = input("Entrez la 1er phrase\n")
        sentence2 = input("Entrez la 2eme phrase\n")
        
        # Make the Preprocessing
        sentence1 = Preprocessing(sentence1)
        sentence2 = Preprocessing(sentence2)
        
        # Encode embeddings of inputs
        emb1 = model.encode(sentence1)
        emb2 = model.encode(sentence2)
        
        # Compute the cosinus similarity
        sim = cosine_similarity(emb1.reshape(1, -1),emb2.reshape(1, -1))
        print("Similarité : " + str(sim[0]))
        
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
            saveSample(file,sentence1,sentence2,score,str(sim[0][0]))
        
        # Ask user if he wants to continue or not
        stop_in = input("Voulez vous stopper ? o ou n\n")
        if stop_in == "o":
            stop = False