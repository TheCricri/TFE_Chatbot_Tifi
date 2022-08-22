"""
TFE - Chatbot Tifi - Technifutur
by Nicolas Christiaens
"""

from Interface import Interface
from Chatbot import Chatbot

if __name__ == "__main__":
    # Create the chatbot with good arguments
    Tifi = Chatbot(name = "TIFI",
                   logsFile = "Chatbot_Logs.xlsx",
                   QA_file = "QuestionsReponses.xlsx",
                   MyIntentsClassificationModel = "./Model_IntentsClassification/Final_model/",
                   SentenceEmbeddingsModel = "Model_SentenceEmbedding/Finetuning/Final_model")
    
    # Create the user interface object
    Int = Interface(Tifi)