"""
TFE - Chatbot Tifi - Technifutur
by Nicolas Christiaens
"""

from ScrapFormationsDB import processAll,readDF,extractContacts,extractPlanning
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
import torch
from Preprocessing import Preprocessing

# Class to handle the part of the chatbot which handle the training Database of Technifutur
class Chatbot_formation():
    def __init__(self,model_name):
        # Load the formations database
        self.formationsDB = self.loadDB()
        
        # Load the models
        self.NeedTranslation = False
        if model_name == "sentence-transformers/all-distilroberta-v1":
            self.NeedTranslation = True
            # Model for translation from Hugginface
            self.translation = pipeline("translation_fr_to_en",model="Helsinki-NLP/opus-mt-fr-en")

        self.model = SentenceTransformer(model_name)
        
        # Compute all the embeddings
        self.AllEmbeddings = self.getAllEmbeddings()
    
    # Function to load the training Database (from a file if available or directly from Technifutur)
    def loadDB(self):
        try:
            df = readDF()
        except:
            df = processAll()
        return df

    # Function to separate the Database based on domain (3 sub-Databases possible)
    def getSpecificDB(self,dom):
        if (dom != "ens") & (dom != "ent") & (dom != "de"):
            print("Bad domain")
            return None
        new_index = self.formationsDB.index[self.formationsDB[dom] == "True"]
        new_DB = self.formationsDB.iloc[new_index]
        return new_DB
    
    # Function to extract all the contact of a specific training from the Database
    def getContacts(self,index,dom):
        if (dom != "ens") & (dom != "ent") & (dom != "de"):
            print("Bad domain")
            return None
        contacts = extractContacts(self.formationsDB.iloc[[index]],dom)
        info_contacts = []
        if contacts is None:
            return info_contacts
        for index,row in contacts.iterrows():
            info = [row["prenom"],row["nom"],row["fonction"],row["tel"],row["email"]]
            info_contacts.append(info)
        return info_contacts,dom   
        
    # Function to get the index of a training with its title
    def getIndexFromIntitule(self,intitule):
        index = self.formationsDB.index[self.formationsDB["intitule"] == intitule]
        return index[0]
    
    # Function to get the title of a training with its index
    def getIntituleFromIndex(self,index):
        intitule = self.formationsDB.intitule[index]
        return intitule
    
    # Function to get the idax of a training with its index
    def getIdax(self,index):
        idax = self.formationsDB.idax[index]
        return idax
    
    # Function to get the index of a training with its idax
    def getIndexFromIdax(self,idax):
        index = self.formationsDB.index[self.formationsDB["idax"] == idax]
        return index[0]
    
    # Function to get the timing of a training with its index
    def getTiming(self,index):
        timing = self.formationsDB.duree_txt[index]
        return timing
    
    # Function to get the maximum number of people of a training with its index
    def getNumberOfPeople(self,index):
        people = str(self.formationsDB.participant[index])
        return people
    
    # Function to get the method of a training with its index
    def getMethodo(self,index):
        meth = self.formationsDB.methodo[index]
        return meth
    
    # Function to get the program of a training with its index
    def getProgramme(self,index):
        prog = self.formationsDB.programme[index]
        return prog
    
    # Function to get the prerequisite of a training with its index
    def getPrerequis(self,index):
        pre = self.formationsDB.prerequis[index]
        return pre
    
    # Function to get the objective of a training with its index
    def getObjectif(self,index):
        obj = self.formationsDB.objectif[index]
        return obj
    
    # Function to get the targets of a training with its index
    def getCible(self,index):
        cible = self.formationsDB.cible[index]
        return cible
    
    # Function to get the planning of a training with its index
    def getPlanning(self,index):
        planning = extractPlanning(self.formationsDB.iloc[[index]])
        info_planning = []
        if planning is None:
            return info_planning
        for index,row in planning.iterrows():
            info = [row["cible"],row["premierjour"],row["dernierjour"],row["location"]]
            info_planning.append(info)
        return info_planning  
    
    # Function to return the full name of a domain name
    def returnFullName(self,name):
        if name.lower() == "ent":
            name = "les entreprises"
        if name.lower() == "de":
            name = "les demandeurs d'emplois"
        if name.lower() == "ens":
            name = "l'enseignement"
        return name
    
    # Function to get a message which describes the planning of a training with its index
    def messagePlanning(self,index):
        planning = self.getPlanning(index)
        nb_form = len(planning)
        if nb_form == 0:
            message = "Malheureusement la formation n'a pas encore été planifiée"
        else:
            message = "Voici le(s) planning(s) encodé(s) :"
            for index in range(nb_form):
                message = message + "\n"
                cible = self.returnFullName(planning[index][0])
                message = message + "Du {} au {} pour {} à {}".format(planning[index][1],planning[index][2],cible,planning[index][3])
        return message
    
    # Function to get a message which describes the contacts of a training with its index and the domain
    def messageContacts(self,index,dom):
        contacts,dom = self.getContacts(index,dom)
        nb_cont = len(contacts)
        cible = self.returnFullName(dom)
        if nb_cont == 0:
            message = "Malheureusement aucun contact n'est inscrit pour {} dans cette formation".format(cible)
        else:
            message = "Voici le(s) contact(s) enregistré(s) pour {}:".format(cible)
            for index in range(nb_cont):
                message = message + "\n"
                message = message + "{} {} ({}) rejoignable par téléphone au {} ou par email via {}".format(contacts[index][0],contacts[index][1],contacts[index][2],contacts[index][3],contacts[index][4])
        return message
    
    # Function to get a starting message (title + idax) of a training with its index
    def messageStarting(self,index):
        intitule = self.getIntituleFromIndex(index)
        idax = self.getIdax(index)
        message = "La formation '{}' ({}) ".format(intitule,idax)
        return message
    
    # Function to get a message which describes the targets of a training with its index
    def messageCible(self,index):
        cible = self.getCible(index)
        message = self.messageStarting(index)
        if cible == "":
            message = message + "n'a pas de cible particulière"
        else:
            message = "Informations concernant les personnes ciblées par " + message[0].lower() + message[1:] + ": " + cible
        return message      
                
    # Function to get a message which describes the prerequisite of a training with its index
    def messagePrerequis(self,index):
        pre = self.getPrerequis(index)
        message = self.messageStarting(index)
        if pre == "":
            message = message + "n'a pas de prérequis particulier"
        else:
            message = "Informations concernant les prérequis pour " + message[0].lower() + message[1:] + ": " + pre
        return message  

    # Function to get a message which describes the program of a training with its index
    def messageProgramme(self,index):
        prog = self.getProgramme(index)
        message = self.messageStarting(index)
        if prog == "":
            message = message + "n'a pas de programme encodé"
        else:
            message = "Informations concernant le programme de " + message[0].lower() + message[1:] + ": " + prog
        return message  

    # Function to get a message which describes the method of a training with its index
    def messageMethodo(self,index):
        met = self.getMethodo(index)
        message = self.messageStarting(index)
        if met == "":
            message = message + "n'a pas de méthodo particulière"
        else:
            message = "Informations concernant la méthodo de " + message[0].lower() + message[1:] + ": " + met
        return message  

    # Function to get a message which describes the timing and maximum number of people of a training with its index
    def messageTimingAndPeople(self,index):
        time = self.getTiming(index)
        people = self.getNumberOfPeople(index)
        message = self.messageStarting(index)
        if time == "":
            message = message + "n'a pas d'informations complémentaires concernant le nombre de participant ou la durée"
        else:
            message = "Informations supplémentaires concernant " + message[0].lower() + message[1:] + ": " + " Il y aura " + people + " participants et elle durera " + time[0].lower() + time[1:]
        return message            
                
    # Function to make search in the Database (title) and return a certain number of best matches and that can be depend of a specific domain (3 possible)
    def makeGlobalResearch(self,msg,number,dom=None):
        # Check if translation is needed
        if self.NeedTranslation is True:
            msg = self.translation(msg)[0]["translation_text"]
        
        # Get the good sub-Database if dom is not None
        if dom is not None:
            df = self.getSpecificDB(dom)
            AllIndex = df.index
        else:
            AllIndex = self.formationsDB.index
            
        # Get the embedding of the message
        with torch.no_grad():
            currentEmb = self.model.encode(msg)
            
        # Compute the cosinus similarities of all the embeddings of the titles of the DB with the embedding of the message
        similarities = cosine_similarity(currentEmb.reshape(1,-1),self.AllEmbeddings[AllIndex])
        
        # Best score
        pos = similarities.argmax()
        best_score = similarities[0][pos]
        print(best_score)
        
        # Return a message with the requested number of best training matches
        formations = ""
        for i in range(number):
            pos = similarities.argmax()
            score = similarities[0][pos]
            similarities[0][pos] = 0
            index = AllIndex[pos]
            formation_found = self.messageStarting(index)
            formations = formations + formation_found + " --> " + str(score) +"\n"
        return formations,best_score
                
    # Function to search in the Database a training (title + idax) with its Idax. This function is used to see if a specific Idax exists or not
    def makeIdaxResearch(self,idax):
        try:
            index = self.getIndexFromIdax(idax)
            response = self.messageStarting(index)
            return response
        except:
            return None
    
    # Function to compute the embedding of all the title in the Database
    def getAllEmbeddings(self):
        # Get all the titles of formations
        intitules = self.formationsDB["intitule"].tolist()
        
        # Make the preprocessing
        for i in range(len(intitules)):
            intitules[i] = Preprocessing(intitules[i])
        
        # Check if translation is needed
        if self.NeedTranslation is True:
            for i in range(len(intitules)):
                intitules[i] = self.translation(intitules[i])[0]["translation_text"]
        # Get the embeddings
        with torch.no_grad():
            result = self.model.encode(intitules)
        return result
    
