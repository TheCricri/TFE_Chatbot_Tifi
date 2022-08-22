"""
TFE - Chatbot Tifi - Technifutur
by Nicolas Christiaens
"""

import pandas as pd
from FineTuning import mean_pooling
from transformers import AutoModel,AutoTokenizer
import torch
from transformers import pipeline
from Preprocessing import Preprocessing

# Class to create the part of the chatbot which handle the Qestion-Answer Database that we have created
class Chatbot_QA():
    def __init__(self,QA_file,model_name):
        # Process QA file
        self.QA_file = QA_file
        self.getData()
        
        # Load the translation model if needed
        self.NeedTranslation = False
        if model_name == "sentence-transformers/all-distilroberta-v1":
            self.NeedTranslation = True
            # Model for translation from Hugginface
            self.translation = pipeline("translation_fr_to_en",model="Helsinki-NLP/opus-mt-fr-en")
        
        # Load the sentence embedding model
        self.max_length = 128
        self.model = AutoModel.from_pretrained(model_name,max_length=self.max_length)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)    
        
        # Get the embeddings from all the possible questions
        self.allEmbeddings = self.getAllEmbeddings()
    
    # Function to get the data from the file
    def getData(self):
        self.data = pd.read_excel("QuestionsReponses.xlsx",index_col=None, header=0)
        # Apply the preprocessing
        self.data["Question"] = self.data["Question"].apply(Preprocessing)
    
    # Function to get all the embeddings of the questions from the data
    def getAllEmbeddings(self):
        # Get all the encoded questions
        questions = self.data["Question"].tolist()
        
        # Translate if needed
        if self.NeedTranslation is True:
            for i in range(len(questions)):
                questions[i] = self.translation(questions[i])[0]["translation_text"]
        
        # Get embedding (mean pooling + normalization)
        tokens = self.tokenizer(questions,padding=True,truncation=True,max_length=self.max_length,return_tensors="pt")
        with torch.no_grad():
            h = self.model(**tokens)
        emb = mean_pooling(h,tokens["attention_mask"],normalize=True)
        return emb
    
    # Function to process a message and return the best response in our QA Database with the score
    def QAencoder(self,msg):
        # Check if translation is needed
        if self.NeedTranslation is True:
            msg = self.translation(msg)[0]["translation_text"]
        
        # Get the embedding of the message
        tokens = self.tokenizer([msg],padding=True,truncation=True,max_length=self.max_length,return_tensors="pt")
        with torch.no_grad():
            h = self.model(**tokens)
        emb = mean_pooling(h,tokens["attention_mask"],normalize=True)
    
        # Compute all cosinus similarity of the embeddings of the questions from the data with the embedding of the message
        values = torch.cosine_similarity(emb,self.allEmbeddings)
        values = values.numpy()
        
        # Get the best score
        pos = values.argmax()
        score = values[pos]
        
        # Get the response corresponding to the best score
        response = self.data["Réponse"][pos]
        
        return response,score