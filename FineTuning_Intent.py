"""
TFE - Chatbot Tifi - Technifutur
by Nicolas Christiaens
"""

import pandas as pd
import torch
from datasets import load_dataset
from transformers import AdamW,get_linear_schedule_with_warmup
from Preprocessing import Preprocessing
import matplotlib.pyplot as plt
from transformers import AutoModelForSequenceClassification,AutoTokenizer,Trainer,pipeline
from torch.utils.data import DataLoader
import math
from tqdm.auto import tqdm
from datasets import load_metric
from sklearn.metrics import confusion_matrix,classification_report
import numpy as np
import seaborn as sn

# Function to load the Dataset
def load_IntentsDS(path="IntentsDS.csv"):
    intents = load_dataset("csv", data_files=path,names=["text", "label"])
    return intents

# Function to load the intent Dataset description
def load_IntentsDescription():
    description = pd.read_excel("IntentsDescription.xlsx",header=0)
    return description

# Function to load the Dataset and make preprocessing
def loadAndProcess(path="IntentsDS.csv",translate=False):
    intents = pd.read_csv(path,names=["text", "label"])
    intents["text"] = intents["text"].apply(Preprocessing)
    
    ## If dataset translation needed
    if translate is True:
        # Model for translation from Hugginface
        translation = pipeline("translation_fr_to_en",model="Helsinki-NLP/opus-mt-fr-en")
    
        # Function to translate French in English
        def translate(text):
            new_text = translation(text)[0]["translation_text"]
            return new_text
        
        # Make the translation
        intents["text"] = intents["text"].apply(translate)
        
    intents.to_csv("TIntentsDS_WithPreprocessing.csv",header=False,index=False)


if __name__ == "__main__":
    # Inform the user if no GPU is detected
    if torch.cuda.is_available() is True:
        device = "cuda"
    else:
        print("Pas de GPU pour le training")
    
    # # Set the seed
    # torch.manual_seed(3)
    # torch.cuda.manual_seed_all(3)
    
    # Make preprocessing on the dataset
    # loadAndProcess()
    
    # Load data
    intents = load_IntentsDS(path="TIntentsDS_WithPreprocessing.csv")
    descriptions = load_IntentsDescription()
    
    # Function to translate the label to intent name
    def labelToName(label):
        intent = descriptions[descriptions["Label"] == label]["Intents"]
        intent = intent.item()
        return intent
    
    # # Analyse the dataset and make the preprocessing
    # intents.set_format(type="pandas")
    # analyse = intents["train"][:]
    # # Create the intent name value
    # analyse["intent_name"] = analyse["label"].apply(labelToName)
    # # Plot the length of the different intents
    # analyse["numberOfWord"] = analyse["text"].str.split().apply(len)
    # analyse.boxplot("numberOfWord",by="intent_name",grid=False,color="black",vert=False)
    # plt.suptitle("")
    # plt.title("Number of words")
    # plt.xlabel("")
    # plt.show()
    # # Reset the format for training the model
    # intents.reset_format()
    
    # Define the number of different labels (output)
    num_labels = len(descriptions)
    
    # # Huggingface path for all models
    # CamemBERT : camembert-base
    # FlauBERT : flaubert/flaubert_base_uncased
    # DistilCamembert : cmarkea/distilcamembert-base
    # BERT Multilingual : bert-base-multilingual-uncased
    # DistilRoberta (English) : distilroberta-base
    
    # Create the tokenizer and classification model with Huggingface automatic library
    model_name = "camembert-base"
    model_save = "Model_IntentsClassification/test"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name,num_labels=num_labels,max_length=128).to(device)
    
    # Define tokenize function
    def tokenize(df):
        return tokenizer(df["text"],padding=True,truncation=True,max_length=128)
    
    # Tokenize the Dataset
    intents_encoded = intents["train"].map(tokenize,batched=True,batch_size=None)
    
    # Transform the dataset in the correct form for pytorch
    intents_encoded = intents_encoded.remove_columns(["text"])
    intents_encoded = intents_encoded.rename_column("label", "labels")
    
    # Create the differents splits : train - validation - test and keep them balanced
    seed = 3
    intents_encoded = intents_encoded.class_encode_column("labels")
    splits = intents_encoded.train_test_split(test_size=8/40,stratify_by_column="labels",seed=seed)
    trainANDvalid = splits["train"]
    test = splits["test"]
    splits2 = trainANDvalid.train_test_split(test_size=8/32,stratify_by_column="labels",seed=seed)
    train = splits2["train"]
    validation = splits2["test"]
    
    # Make the datasets working with pytorch framework
    train.set_format("torch")
    validation.set_format("torch")
    test.set_format("torch")
    trainANDvalid.set_format("torch")
    
    # Set the training parameters 
    warmup = 0.06
    batch_size = 32
    learning_rate = 6e-5
    weight_decay = 0.01
    num_epochs = 100
    break_epoch = 47
    
    # # Parameters for best CamemBERT model
    # warmup = 0.06
    # batch_size = 32
    # learning_rate = 6e-5
    # weight_decay = 0.01
    # num_epochs = 100
    # break_epoch = 47
    
    # # Paremeters for best FlauBERT model
    # warmup = 0.06
    # batch_size = 16
    # learning_rate = 4e-5
    # weight_decay = 0.01
    # num_epochs = 150
    # break_epoch = 80
    
    # # Parameters for best DistilCamemBERT model
    # warmup = 0.06
    # batch_size = 64
    # learning_rate = 10e-5
    # weight_decay = 0.01
    # num_epochs = 100
    # break_epoch = 19
    
    # # Parameters for best Bert Multilingual model
    # warmup = 0.06
    # batch_size = 32
    # learning_rate = 5e-5
    # weight_decay = 0.01
    # num_epochs = 100
    # break_epoch = 30
    
    # Create the dataloaders with the different datasets
    #trainloader = DataLoader(train,shuffle=True,batch_size=batch_size)
    trainloader = DataLoader(trainANDvalid,shuffle=True,batch_size=batch_size)
    validationloader = DataLoader(validation,batch_size=batch_size)
    testloader = DataLoader(test,batch_size=batch_size)
    
    # Set the optimizer and scheduler with warm
    optimizer = AdamW(model.parameters(),lr=learning_rate,weight_decay=weight_decay)
    training_steps = num_epochs*len(trainloader)
    warmup_steps = math.ceil(training_steps*warmup)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,num_warmup_steps=warmup_steps,num_training_steps = training_steps)

    # Set up the progress bar
    progress_bar = tqdm(range(training_steps))

    # Train with pytorch framework and set variables for keeping information
    loss_values_train = []
    loss_values_validation = []
    min_loss = 9999999
    min_epoch = 0
    for epoch in range(num_epochs):
        # Train with the training set
        model.train()
        # Create the loss variable for each epoch
        tmp_loss = []
        for batch in trainloader:
            # Clear the gradient
            optimizer.zero_grad()
            # Batch to GPU
            batch = {k: v.to(device) for k, v in batch.items()}
            # Predict the batch
            outputs = model(**batch)
            # Get the loss
            loss = outputs.loss
            # Compute the gradient
            loss.backward()
            # Make the step of training
            optimizer.step()
            scheduler.step()
            # Update the progess bar
            progress_bar.update(1)
            # Add the loss
            tmp_loss.append(loss.item())
        # Make the loss independant to the batch size
        tmp_loss = sum(tmp_loss)/len(trainloader)
        # Append the epoch training loss
        loss_values_train.append(tmp_loss)
        
        # Evaluate on the validation set
        tmp_loss = []
        model.eval()
        for batch in testloader:
            # Batch to GPU
            batch = {k: v.to(device) for k, v in batch.items()}
            # Predict the batch (no gradients needed)
            with torch.no_grad():
                outputs = model(**batch)
            # Get the loss
            loss = outputs.loss
            # Append the loss
            tmp_loss.append(loss.item())
        # Make the loss independant to the batch size
        tmp_loss = sum(tmp_loss)/len(validationloader)
        # Save the minimum loss and epoch
        if tmp_loss < min_loss:
            min_loss = tmp_loss
            min_epoch = epoch
        # Append the epoch validation loss
        loss_values_validation.append(tmp_loss)
        
        # Manual break
        if epoch == break_epoch:
            break
     
    # Plot the 2 losses
    plt.plot(loss_values_train,"r",label="train set")
    plt.plot(loss_values_validation,"b",label = "validation set")
    plt.title("Training of model Base for Intents Classification")
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    
    # Evaluate on validation or on test set 
    accuracy = load_metric("accuracy")
    precision = load_metric("precision")
    recall = load_metric("recall")
    f1 = load_metric("f1")
    for batch in testloader:
        # Batch to GPU
        batch = {k: v.to(device) for k, v in batch.items()}
        # Predict the batch
        with torch.no_grad():
            outputs = model(**batch)
        # Get the logit vectors
        logits = outputs.logits
        # Get the best predictions
        predictions = torch.argmax(logits,dim=-1)
        
        # Add the predictions and the true labels to the metric calculators
        accuracy.add_batch(predictions=predictions, references=batch["labels"])
        precision.add_batch(predictions=predictions, references=batch["labels"])
        recall.add_batch(predictions=predictions, references=batch["labels"])
        f1.add_batch(predictions=predictions, references=batch["labels"])
    
    # Save the model
    trainer = Trainer(model=model,tokenizer=tokenizer)
    trainer.save_model(model_save)
    
    # Print results
    print("Min at epoch " + str(min_epoch) +" : " + str(min_loss))
    print("Accuracy : "  + str(accuracy.compute()))
    print("Precision : " + str(precision.compute(average="macro")))
    print("Recall : " + str(recall.compute(average="macro")))
    print("F1 : " + str(f1.compute(average="macro")))
    
    # Create the confusion matrix
    y_pred = []
    y_true = []
    for batch in testloader:
        # Batch to GPU
        batch = {k: v.to(device) for k, v in batch.items()}
        # Predict the batch
        with torch.no_grad():
            outputs = model(**batch)
        # Get the logit vectors
        logits = outputs.logits
        # Get the best predictions
        predictions = torch.argmax(logits,dim=-1)
        
        # Get the predictions
        cpu_pred = predictions.cpu().numpy()
        for pred in cpu_pred:
            y_pred.append(pred)
            
        # Get the true labels
        cpu_true = batch["labels"].cpu().numpy()
        for true in cpu_true:
            y_true.append(true)
     
    # Get the intent names
    intent_names = descriptions["Intents"].tolist()
        
    # Code to construct the confusion matrix (from https://christianbernecker.medium.com/how-to-create-a-confusion-matrix-in-pytorch-38d06a7f04b7)
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix)*num_labels,index=[i for i in intent_names],columns=[i for i in intent_names])
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    plt.show()
    
    # Print classification report to well understand
    print(classification_report(y_true, y_pred, target_names=intent_names))