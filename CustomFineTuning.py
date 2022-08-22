"""
TFE - Chatbot Tifi - Technifutur
by Nicolas Christiaens
"""

from datasets import Dataset
from FineTuning import STSBTrainingModel
from torch.utils.data import DataLoader
import pandas as pd
from Preprocessing import Preprocessing
from transformers import AdamW,get_constant_schedule
from transformers import AutoModel,AutoTokenizer,Trainer
import torch
from tqdm.auto import tqdm

# Load the custom dataset and make our preprocessing
def getCustomDS(file="customDS.xlsx"):
    df = pd.read_excel(file)
    
    df["sentence1"] = df["sentence1"].apply(Preprocessing)
    df["sentence2"] = df["sentence2"].apply(Preprocessing)
    df["score"] = df["score"].astype(float)
    
    dataset = Dataset.from_pandas(df)
    
    return dataset
    

if __name__ == "__main__":
    # Inform the user if no GPU is detected
    if torch.cuda.is_available() is True:
        device = "cuda"
    else:
        print("Pas de GPU pour le training")
        
    # Read the custom dataset
    train = getCustomDS()

    # Set Global Parameters
    max_length = 128
    model_name = "Model_SentenceEmbedding/Finetuning/Final_model"
    model_save = "Model_SentenceEmbedding/Custom/Final_model"
    batch_size = 16
    learning_rate = 2e-5
    weight_decay = 0.01
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    num_epochs = 2

    # Create the tokenize function
    def tokenize1(df):
        return tokenizer(df["sentence1"],padding=True,truncation=True,max_length=max_length)
    
    def tokenize2(df):
        return tokenizer(df["sentence2"],padding=True,truncation=True,max_length=max_length)

    # Transform in the correct form : ['input_ids1', 'attention_mask1', 'input_ids2', 'attention_mask2','score']
    train_encoded = train.map(tokenize1,batched=True,batch_size=None)
    train_encoded = train_encoded.rename_column("input_ids","input_ids1")
    train_encoded = train_encoded.rename_column("attention_mask","attention_mask1")
    train_encoded = train_encoded.map(tokenize2,batched=True,batch_size=None)
    train_encoded = train_encoded.rename_column("input_ids","input_ids2")
    train_encoded = train_encoded.rename_column("attention_mask","attention_mask2")
    train_encoded = train_encoded.remove_columns(["sentence1"])
    train_encoded = train_encoded.remove_columns(["sentence2"])
    train_encoded = train_encoded.remove_columns(["Old Similarity"])
    
    # Set the correct format
    train_encoded.set_format("torch")
    
    # Create the Dataloader
    trainloader = DataLoader(train_encoded,shuffle=True,batch_size=batch_size)

    # Load the model used as body
    body = AutoModel.from_pretrained(model_name,max_length=max_length)

    # Create the training model
    model = STSBTrainingModel(body=body).to(device)
    
    # Load the model and it
    optimizer = AdamW(model.parameters(),lr=learning_rate,weight_decay=weight_decay)
    training_steps = num_epochs*len(trainloader)
    scheduler = get_constant_schedule(optimizer=optimizer)

    # Set up the progress bar
    progress_bar = tqdm(range(training_steps))

    # Loss keeper
    loss_train = []
    
    # Get the loss without training (epoch 0)
    tmp_loss = []
    for batch in trainloader:
        # Batch to GPU
        batch = {k: v.to(device) for k, v in batch.items()}
        # Predict the batch (no gradients needed)
        with torch.no_grad():
            loss,_ = model(**batch)
        # Append the loss
        tmp_loss.append(loss.item())
    # Make the loss independant to the batch size
    tmp_loss = sum(tmp_loss)/len(trainloader)
    # Append the epoch training loss
    loss_train.append(tmp_loss)

    # Train the model
    for epoch in range(num_epochs):
        model.train()
        tmp_loss = []
        for batch in trainloader:
            # Clear the gradient
            optimizer.zero_grad()
            # Batch to GPU
            batch = {k: v.to(device) for k, v in batch.items()}
            # Predict the batch
            loss,_ = model(**batch)
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
        loss_train.append(tmp_loss)

    # Save the trained model with the tokenizer
    trainer = Trainer(model=body,tokenizer=tokenizer)
    trainer.save_model(model_save)
