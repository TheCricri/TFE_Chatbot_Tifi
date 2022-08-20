"""
TFE - Chatbot Tifi - Technifutur
by Nicolas Christiaens
"""

import pandas as pd
from Chatbot_QA import Chatbot_QA
from Chatbot_formation import Chatbot_formation
import time
from openpyxl import load_workbook
from TextScripts import *
from random import choice,seed
from transformers import AutoTokenizer,AutoModelForSequenceClassification,TextClassificationPipeline
import torch
import re
from Preprocessing import Preprocessing
from FineTuning_Intent import load_IntentsDescription

class Chatbot():
    def __init__(self,name,logsFile,QA_file,MyIntentsClassificationModel,SentenceEmbeddingsModel):
        self.name = name
        self.logsFile = logsFile
        self.createLogs()
        
        # Create all the different composents for responding
        self.chatbot_QA = Chatbot_QA(QA_file,SentenceEmbeddingsModel)
        self.model_intents = AutoModelForSequenceClassification.from_pretrained(MyIntentsClassificationModel)
        self.tokenizer_intents = AutoTokenizer.from_pretrained(MyIntentsClassificationModel)
        self.intents_classifier = TextClassificationPipeline(model=self.model_intents,tokenizer=self.tokenizer_intents)
        self.chatbot_formation = Chatbot_formation(SentenceEmbeddingsModel)
        
        # Constant value
        self.intents_description = load_IntentsDescription()
        
        # Variables
        self.repeat = False
        self.currentIdax = None
        self.IdaxInMemory = None
        self.IsAskingForInscription = None
        self.AskingProfilInscription = None
        self.AskingProfilResearch = None
        self.userProfil = None
        self.IsAskingForInformation = None
        self.IsAskingForInformationQuery = None
        self.lastCorrectIdax = None
        seed(0)
        
    # Function to get the response from a user message (main process of the chatbot class)
    def getResponse(self,msg):
        # Keep the initial message for logs
        saved = msg
        
        # Search for idax in the message before preprocessing
        self.currentIdax = self.searchForIdaxPattern(msg)
        
        # Delete the idax if finded and save last Idax found
        if self.currentIdax is not None:
            msg = msg.replace(self.currentIdax,"")
            # Test the idax
            test = self.chatbot_formation.makeIdaxResearch(self.currentIdax)
            if test is not None:
                self.lastCorrectIdax = self.currentIdax
        
        # Make preprocessing on the message
        msg = Preprocessing(msg)
  
        # Search the user profil for inscription (Flag condition)
        if self.AskingProfilInscription is not None:
            # Check if the user gives his profil for completing the current task or not
            profil = self.searchForProfil(msg)
            if profil is not None:
                self.userProfil = profil
                response = self.AskingForInscriptionHandler(self.AskingProfilInscription)
            else:
                response = choice(AskingProfilInscriptionError)
            self.AskingProfilInscription = None
            self.saveLogs(saved,response,"-","Formations DB - AskingProfilInscription")
            return response
            
        # Search the user profil for training research (Flag condition)
        if self.AskingProfilResearch is not None:
            self.userProfil = self.searchForProfil(msg)
            formations_found,best_score = self.chatbot_formation.makeGlobalResearch(self.AskingProfilResearch,5,self.userProfil)
            response = choice(FormationOfferMessages) + formations_found
            self.AskingProfilResearch = None
            self.saveLogs(saved,response,"-","Formations DB - AskingProfilResearch")
            return response
        
        # Handle when the user ask information with the idax (Flag condition)
        if self.IdaxInMemory is not None:
            # Check if the user gives a correct information for completing the current task or not
            info = self.searchForInformation(msg)
            if info is not None:
                response = self.InformationRetriever(info,self.IdaxInMemory)
            else:
                response = choice(InformationResearchError)
            self.IdaxInMemory = None
            self.saveLogs(saved,response,"-","Formations DB - IdaxInMemory")
            return response
        
        # Get the score and the possible response from our Question-Answer (QA) database
        response,score_QA = self.chatbot_QA.QAencoder(msg)
        print(score_QA)
        # Set thresholds for QA and intents score
        t_QA = 0.72
        t_intents = 0.6
        
        # Main process for responding according to the scores
        if score_QA < t_QA:
            # If no good answer found in the QA database, we check the intents
            intent,score_intents = self.IntentsClassification(msg)
            print(score_intents)
            if score_intents > t_intents:
                # If a intent is found with a good score
                score = score_intents
                response = self.IntentsHandler(intent,msg)
                source = "Formations DB"
                self.repeat = False
            else:
                # If there is no good intent, we use script message which depend on the number of unclassified messages in a row
                score = score_intents
                source = "Script Message"
                if self.repeat is True:
                    response = choice(ErrorMessages)
                    self.repeat = False
                else:
                    response = choice(RepeatMessages)
                    self.repeat = True
        else:
            # If there is a good answer found in the QA databse
            score = score_QA
            self.repeat = False
            source = "QA DB"
            
        # Save logs and return the response
        self.saveLogs(saved,response,score,source)
        return response  
        
    # Function to create the logs file
    def createLogs(self):
        try:
            open(self.logsFile)
        except:
            df = pd.DataFrame(columns=['Time','Message','Response','Score','Source'])
            df.to_excel(self.logsFile,index=False)
    
    # Function to get the name of the chatbot
    def getName(self):
        return self.name
    
    # Function to get the welcome message
    def getWelcomeMessage(self):
        message = choice(WelcomeMessages)
        return message
    
    # Function to save the logs of a conversation
    def saveLogs(self,msg,response,score,source):
        # Get the time
        time_msg = time.time()
        wmdhy = time.ctime(time_msg)
        
        # Get the log
        new_log = [wmdhy,msg,response,score,source]
        
        # Save the log
        wb = load_workbook(self.logsFile)
        page = wb.active
        page.append(new_log)
        wb.save(filename=self.logsFile)

    # Function to reset all the current variables
    def resetState(self):      
        self.repeat = False
        self.currentIdax = None
        self.IdaxInMemory = None
        self.IsAskingForInscription = None
        self.AskingProfilInscription = None
        self.AskingProfilResearch = None
        self.userProfil = None
        self.IsAskingForInformation = None
        self.IsAskingForInformationQuery = None
        self.lastCorrectIdax = None
      
    # Function to get the best intent and the score of this intent from a message
    def IntentsClassification(self,msg):
        with torch.no_grad():
            result = self.intents_classifier(msg)
        intent = self.predictionTranslater(result)
        score = result[0]["score"]
        return intent,score
    
    # Function to translate the output of the model into a Intent (string)
    def predictionTranslater(self,pred):
        label = pred[0]["label"]
        number = re.findall(r"\d+",label)
        number = int(number[0])
        intent = self.intents_description[self.intents_description["Label"] == number]["Intents"]
        intent = intent.item()
        return intent
    
    # Function to handle all the different intents with specific actions and responses
    def IntentsHandler(self,intent,msg):
        # Preventive message in case of misuse of the prediction model (should not appear but prevent the program from crashing due to an error)
        response = "Erreur dans les intentions"
        print(intent)
        # The user asks if there are training available in a certain field or for a certain software
        if intent == "AskForFormation":
            # Check if the word formation is present to improve the search
            result = self.searchAfterFormation(msg)
            # Check and ask for the user profil
            if self.userProfil is None:
                profil = self.searchForProfil(msg)
                self.userProfil = profil
            # If the profil of the user is not known, we activate the Flag and ask the user
            if self.userProfil is None:
                response = choice(AskForProfilMessages)
                # Save some results in the Flag
                if result is None:
                    self.AskingProfilResearch = msg
                else:
                    self.AskingProfilResearch = result
            # If the profil of the user is known, we can make the search of training
            else:
                if result is None:
                    result = msg
                # We set the number of training returned to 5
                formations_found,best_score = self.chatbot_formation.makeGlobalResearch(result,5,self.userProfil)
                response = choice(FormationOfferMessages) + formations_found
        
        # The user asks for more information (programs, prerequisites, etc.) concerning a particular training
        if intent == "AskForMoreInformation":
            # Search for keywords about information needed
            info = self.searchForInformation(msg)
            # Get the found Idax
            result = self.currentIdax
            # If no Idax found, we have to search the best matching training and ask the user if it's the training he was looking for
            if result is None:
                # Check if the word formation is present to improve the search
                result2 = self.searchAfterFormation(msg)
                if result2 is None:
                    result = msg
                else:
                    result = result2
                # Get the best matching training
                formation_found,best_score = self.chatbot_formation.makeGlobalResearch(result,number=1)
                # If the search is bad, we take the last correct idax found
                
                response = choice(ConfirmationMessages) + formation_found[0].lower() + formation_found[1:-1] + "?"
                # Set the Flags with some results needed for the next phase
                self.IsAskingForInformation = self.searchForIdaxPattern(formation_found)
                self.IsAskingForInformationQuery = info
            # If a Idax is found we can skip the confirmation phase
            else:
                # Make a search with the Idax
                formation_found = self.chatbot_formation.makeIdaxResearch(result)
                if formation_found is None:
                    response = choice(BadIdaxMessages)
                else:
                    # If the Idax is correct, we can start to retrieve information about the training
                    if info is None:
                        # Set a Flag with the Idax if no information keywords were found in the message
                        self.IdaxInMemory = result
                        response = choice(AskInformationMessages)
                    else:
                        response = self.InformationRetriever(info,result)
        
        # The user asks to register for a training (if possible)
        if intent == "AskForInscription":
            # Get the found Idax
            result = self.currentIdax
            # If no Idax found, we have to search the best matching training and ask the user if it's the training he was looking for
            if result is None:
                # Check if the word formation is present to improve the search
                result2 = self.searchAfterFormation(msg)
                if result2 is None:
                    result = msg
                else:
                    result = result2
                # Get the best matching training
                formation_found,best_score = self.chatbot_formation.makeGlobalResearch(result,number=1)
                # If the search is bad, we take the last correct idax found
                
                response = choice(ConfirmationMessages) + formation_found[0].lower() + formation_found[1:-1] + "?"
                # Set the Flag with the Idax for the next phase
                self.IsAskingForInscription = self.searchForIdaxPattern(formation_found)
            # If a Idax is found we can skip the confirmation phase
            else:
                # Make a search with the Idax
                formation_found = self.chatbot_formation.makeIdaxResearch(result)
                if formation_found is None:
                    response = choice(BadIdaxMessages)
                else:
                    # If the Idax is correct, we can start retrieve the contacts of the training
                    response = self.AskingForInscriptionHandler(result)
        
        # The user says hello to TIFI or asks for help
        if intent == "Greetings":
            response = choice(WelcomeMessages)
        
        # The user has finished his task and leaves the terminal
        if intent == "Goodbye":
            self.resetState()
            response = choice(EndMessages)
        
        # The user asks tifi for its usefulness
        if intent == "AskAboutTIFI":
            response = choice(MyFeaturesMessages)
        
        # The user gives a positive opinion
        if intent == "Yes":
            # If the Flag for the intent "AskForMoreInformation" is up, we can go to the next step
            if self.IsAskingForInformation is not None:
                # Save the last idax found
                self.lastCorrectIdax = self.IsAskingForInformation
                # If no keyword information in the Flag, we ask user for one and set the next Flag
                if self.IsAskingForInformationQuery is None:
                    response = choice(AskInformationMessages)
                    self.IdaxInMemory = self.IsAskingForInformation
                # Else we retrieve the specific information
                else:
                    response = self.InformationRetriever(self.IsAskingForInformationQuery,self.IsAskingForInformation)
                # Set the used Flags to None
                self.IsAskingForInformationQuery = None    
                self.IsAskingForInformation = None
            # If the Flag for the intent "AskForInscription" is up, we can go to the next step
            elif self.IsAskingForInscription is not None:
                # Save the last idax found
                self.lastCorrectIdax = self.IsAskingForInscription
                # We retrieve the contacts for the training
                response = self.AskingForInscriptionHandler(self.IsAskingForInscription)
                # Set the used Flag to None
                self.IsAskingForInscription = None
            # If no Flag are activated, the user says No for no apparent reason so we just response with a script message
            else:
                response = choice(NoLogicalIntentionMessages)
        
        # The user gives a negative opinion
        if intent == "No":
            # If the Flag for the intent "AskForMoreInformation" is up, we stop the task
            if self.IsAskingForInformation is not None:
                response = choice(BadTrainingMessages)
                # Set the used Flags to None
                self.IsAskingForInformationQuery = None
                self.IsAskingForInformation = None
            # If the Flag for the intent "AskForInscription" is up, we stop the task
            elif self.IsAskingForInscription is not None:
                response = choice(BadTrainingMessages)
                # Set the used Flag to None
                self.IsAskingForInscription = None
            # If no Flag are activated, the user says No for no apparent reason so we just response with a script message
            else:
                response = choice(NoLogicalIntentionMessages)
        
        # The user thanks Tifi
        if intent == "Thanks":
            response = choice(ThanksMessages)
            
        return response
    
    # Function to search the word 'formation' and return the words after it
    def searchAfterFormation(self,msg):
        p = re.compile('formation (.*)')
        result = p.findall(msg)
        if len(result) == 0:
            return None
        if len(result[0]) < 6:
            return None
        return result[0]
    
    # Function to search Idax (example : AAAA-BB11-2C2)
    def searchForIdaxPattern(self,msg):
        p = re.compile(r'[A-Z]{3,4}\-[A-Z0-9]{3,4}\-[A-Z0-9]{3,4}')
        result = p.findall(msg)
        if len(result) == 0:
            return None
        return result[0]
    
    # Function to search keywords about information
    def searchForInformation(self,msg):
        if any(w in msg for w in key_programme):
            return "programme"
        if any(w in msg for w in key_methodo):
            return "methodo"
        if any(w in msg for w in key_dureeAndPersonne):
            return "dureeAndPersonne"
        if any(w in msg for w in key_prerequis):
            return "prerequis"
        if any(w in msg for w in key_cible):
            return "cible"
        if any(w in msg for w in key_planning):
            return "planning"
        return None
    
    # Function to collect information from the Chatbot_Formation API
    def InformationRetriever(self,info,idax):
        index = self.chatbot_formation.getIndexFromIdax(idax)
        if info == "programme":
            response = self.chatbot_formation.messageProgramme(index)
        if info == "methodo":
            response = self.chatbot_formation.messageMethodo(index)
        if info == "dureeAndPersonne":
            response = self.chatbot_formation.messageTimingAndPeople(index)
        if info == "prerequis":
            response = self.chatbot_formation.messagePrerequis(index)
        if info == "cible":
            response = self.chatbot_formation.messageCible(index)
        if info == "planning":
            response = self.chatbot_formation.messagePlanning(index)
        return response
    
    # Function to search the profil of the user
    def searchForProfil(self,msg):
        if any(w in msg for w in key_de):
            return "de"
        if any(w in msg for w in key_ent):
            return "ent"
        if any(w in msg for w in key_ens):
            return "ens"
        return None
    
    # Function to handle the task of giving information about inscription for a specific training
    def AskingForInscriptionHandler(self,idax):
        # If no profil, we ask the user and set the Flag
        if self.userProfil is None:
            response = choice(AskForProfilMessages)
            self.AskingProfilInscription = idax
        # If the user's profil is already fixed, we retrieve the contacts for the training
        else:
            index = self.chatbot_formation.getIndexFromIdax(idax)
            response = self.chatbot_formation.messageContacts(index,self.userProfil)
        return response
        