"""
TFE - Chatbot Tifi - Technifutur
by Nicolas Christiaens
"""

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import pipeline
from Preprocessing import Preprocessing

# Class to create the part of the chatbot which handle the Qestion-Answer Database that we have created
class Chatbot_QA():
    def __init__(self,QA_file,model_name):
        # Process QA file
        self.QA_file = QA_file
        self.getData()
        
        # Load the modelself.NeedTranslation = False
        if model_name == "sentence-transformers/all-distilroberta-v1":
            self.NeedTranslation = True
            # Model for translation from Hugginface
            self.translation = pipeline("translation_fr_to_en",model="Helsinki-NLP/opus-mt-fr-en")
            
        self.model = SentenceTransformer(model_name)
        
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
        
        with torch.no_grad():
            result = self.model.encode(questions)
        return result
    
    # Function to process a message and return the best response in our QA Database with the score
    def QAencoder(self,msg):
        # Check if translation is needed
        if self.NeedTranslation is True:
            msg = self.translation(msg)[0]["translation_text"]
        
        # Get the embedding of the message
        with torch.no_grad():
            embedding = self.model.encode(msg)
            
        # Compute all cosinus similarity of the embeddings of the questions from the data with the embedding of the message
        values = cosine_similarity(embedding.reshape(1,-1),self.allEmbeddings)
        
        # Get the best score
        pos = values.argmax()
        score = values[0][pos]
        
        # Get the response corresponding to the best score
        response = self.data["Réponse"][pos]
        
        return response,score