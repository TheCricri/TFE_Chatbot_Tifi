"""
TFE - Chatbot Tifi - Technifutur
by Nicolas Christiaens
"""

#
# This file is used as store for all the responses of the chatbot
# We can create as many different messages as you want for the same variable, the final message will be chosen randomly
#

# Message to Welcome the user with a small introduction
WelcomeMessages = ["Bonjour je suis tifi, l’assistant numérique de Technifutur. Vous pouvez me poser des questions sur l'industrie 4.0 et ses composants.",
                  "Bonjour, je m'appelle tifi, le chatbot de Technifutur. Je peux vous aider au sujet de l'industrie 4.0. Que voulez-vous savoir ?",
                  "Bonjour, je me présente, je suis tifi, l'assistant numérique de Technifutur. Je suis là pour répondre à vos questions sur l'industrie 4.0 et le démonstrateur 4.0. Que voulez-vous savoir ?"]

# Message to ask user to repeat if his message is not understand or out of scope
RepeatMessages = ["Veuillez reformuler votre question pour que je sois sur de bien la comprendre",
                  "Je n'ai pas bien compris votre question, pouvez la reformuler svp? ",
                  "Que je comprenne bien, pouvez reformuler la question ?",
                  "Mmmmmh ... Désolé mais je n'ai pas bien compris, savez-vous reformuler la question svp ?",
                  "Pardon, je n'ai pas bien compris. Vous pouvez reformuler votre question svp ?"]

# Message  to handle a bad response from the user when he is asking for contacts of a specific formation
AskingProfilInscriptionError = ["Désolé je n'ai pas compris, veuillez choisir une des 3 propositions ci dessus quand vous effectuez cette demande (demande annulée)"]

# Message to offer formations to the user
FormationOfferMessages = ["Voici quelques formations dans notre catalogue qui pourraient correspondre :\n"]

# Message to handle a bad response from the user when he is asking for more information about a specific formation
InformationResearchError = ["Désole je ne m'attendais pas à cette réponse, je ne peux rien faire (demande annulée)"]

# Message to handle two messages in a row that are not understood or out of scope
ErrorMessages = ["Désolé nous ne pouvons pas répondre à votre requête, n'hésitez pas à demander quelle tâche je suis capable d'effectuer"]

# Message to ask the profil of the user
AskForProfilMessages = ["Quel est votre profil ? Demandeur d emploi, entreprise, enseignement ou tous ?"]

# Message to handle the confirmation phase
ConfirmationMessages = ["Parlez vous bien de "]

# Message to handle a bad Idax
BadIdaxMessages = ["Malheureusement nous ne trouvons pas cet idax dans notre catalogue, veuillez vérifier ou faire une recherche pour trouver l'idax de la formation"]

# Message to ask the user which specific information about a training he wants to know
AskInformationMessages = ["Que voulez vous savoir à propos de cette formation ? Tapez une de ces catégories : programme, methodo, durée, prerequis, cible ou planning"]

# Message to end a conversation with a user
EndMessages = ["J'espère avoir pu vous aider, à bientôt !"]

# Message to describe the features of the chatbot to the user
MyFeaturesMessages = ["Je peux vous aider en cherchant des formations, en vous donnant des informations les concernant ou en vous redirigeant vers les contacts permettant de vous inscrire"]

# Message to handle the fact that the user is not following a logical path
NoLogicalIntentionMessages = ["Je ne suis pas sur de comprendre votre intention"]

# Message to handle the fact that we propose the bad training to the user
BadTrainingMessages = ["Désolé pour ce déconvenu, veuiller vérifier les informations fournies ou faire une recherche pour trouver l'idax de la formation souhaitée"]

# Message when the user thanks the chatbot
ThanksMessages = ["Pas de problème ! Je suis toujours à votre service si vous avez d'autres questions ou recherches"]


#
# Here we will define the different keywords used to find a specific goal in the message of the user
#

# Keywords for "programme"
key_programme = ["programme"]

# Keywords for "methodo"
key_methodo = ["methodo","méthodologie","méthodo","méthode","methode"]

# Keywords for "dureeAndPersonne"
key_dureeAndPersonne = ["durée","nombre d","dure","combien de jour","participant"]

# Keywords for "prerequis"
key_prerequis = ["prérequis","prerequis","qui peut participer","compétences requise"]

# Keywords for "cible"
key_cible = ["cible","cibles","ciblé"]

# Keywords for "planning"
key_planning = ["planning","horaire","date","plannifi"]

# Keywords for "de"
key_de = ["demandeur d emploi","sans emploi","demandeur d'emploi","chômeur","chomeur"]

# Keywords for "ent"
key_ent = ["entreprise"]

# Keywords for "ens"
key_ens = ["enseignement","enseignant"]


#
# Here we will define all the words that the Preprocessing should not change
#

notchangeWord = ["Excel"]