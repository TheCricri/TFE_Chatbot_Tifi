"""
TFE - Chatbot Tifi - Technifutur
by Nicolas Christiaens
"""

from datasets import load_dataset
from transformers import AdamW,get_constant_schedule_with_warmup
from torch.utils.data import DataLoader
from torch import nn
import matplotlib.pyplot as plt
from transformers import AutoModel,AutoTokenizer,Trainer
import torch
import math
import torch.nn.functional as F
from tqdm.auto import tqdm

# Load STSB french part dataset
def getSTSB():
    train_ds = load_dataset("csv", data_files="STSB_train_rescale.csv",names=["sentence1","sentence2", "score"])
    valid_ds = load_dataset("csv", data_files="STSB_valid_rescale.csv",names=["sentence1","sentence2", "score"])
    test_ds = load_dataset("csv", data_files="STSB_test_rescale.csv",names=["sentence1","sentence2", "score"])

    return train_ds["train"],valid_ds["train"],test_ds["train"]

# Rescale the scores of STSB and save them
def rescaleSTSB():
    train_ds = load_dataset("stsb_multi_mt", name="fr", split="train")
    valid_ds = load_dataset("stsb_multi_mt", name="fr", split="dev")
    test_ds = load_dataset("stsb_multi_mt", name="fr", split="test")
    
    train_ds = processSTSB(train_ds)
    valid_ds = processSTSB(valid_ds)
    test_ds = processSTSB(test_ds)
    
    train_ds.to_csv("STSB_train_rescale.csv",header=False,index=False)
    valid_ds.to_csv("STSB_valid_rescale.csv",header=False,index=False)
    test_ds.to_csv("STSB_test_rescale.csv",header=False,index=False)

# Process the STSB dataset to have good inputs format
def processSTSB(dataset):
    new_scores = []
    for df in dataset:
        rescale_score = float(df["similarity_score"])/5.0
        new_scores.append(rescale_score)
    
    # Add the new column and delete the old score
    dataset = dataset.add_column(name="score", column=new_scores)
    dataset = dataset.remove_columns(["similarity_score"])
    
    return dataset

# Load PAWS-X french part dataset
def getPAWS():
    train_ds = load_dataset("paws-x", name="fr", split="train")
    valid_ds = load_dataset("paws-x", name="fr", split="validation")
    test_ds = load_dataset("paws-x", name="fr", split="test")
    
    train_ds = train_ds.remove_columns(["id"])
    valid_ds = valid_ds.remove_columns(["id"])
    test_ds = test_ds.remove_columns(["id"])
    
    return train_ds,valid_ds,test_ds

# Load XNLI french part dataset
def getXNLI():
    train_ds = load_dataset("xnli", name="fr", split="train")
    valid_ds = load_dataset("xnli", name="fr", split="validation")
    test_ds = load_dataset("xnli", name="fr", split="test")
    
    return train_ds,valid_ds,test_ds

# Define mean_pooling (from https://www.sbert.net/examples/applications/computing-embeddings/README.html)
def mean_pooling(model_output,attention_mask,normalize=False):
    # Get all the embeddings
    token_embeddings = model_output[0]
    # Take into accound masks
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    # Sum the embeddings unmasked
    sum_embeddings = torch.sum(token_embeddings*input_mask_expanded,1)
    # Get the total of unmasked embeddings
    sum_mask = torch.clamp(input_mask_expanded.sum(1),min=1e-9)
    # Divide to have the mean of the unmasked embeddings
    sentence_embedding = sum_embeddings / sum_mask
    # Normalize because we will use cosine similarity at inference time
    if normalize is True:
        sentence_embedding = F.normalize(sentence_embedding,p=2, dim=1)
    return sentence_embedding

# Define the model for PAWS-X
class PAWSTrainingModel(nn.Module):
    def __init__(self,body,label=2,d_model=768):
        super(PAWSTrainingModel, self).__init__()
        
        # Variables
        self.label = label
        self.d_model = d_model
        
        # Model used as body
        self.body = body
        
        # Head
        self.head = nn.Linear(3*d_model,label)
    
        # Loss function
        self.loss = nn.CrossEntropyLoss()

    # Define the forward pass
    def forward(self,input_ids1=None,attention_mask1=None,input_ids2=None,attention_mask2=None,labels=None):
        # Get all hidden vectors from the body for the two sentence
        h1 = self.body(input_ids1)
        h2 = self.body(input_ids2)
        
        # Use mean pooling
        u = mean_pooling(h1,attention_mask1)
        v = mean_pooling(h2,attention_mask2)
        
        # Concatenation
        input_concat = []
        input_concat.append(u)
        input_concat.append(v)
        input_concat.append(torch.abs(u-v))
        input_head = torch.cat(input_concat,1)
        
        # Go to the head
        logits = self.head(input_head)
        
        # Compute the loss
        Headloss = self.loss(logits,labels.view(-1))
        
        return Headloss,logits


# Define the model for XNLI
class XNLITrainingModel(nn.Module):
    def __init__(self,body,label=3,d_model=768):
        super(PAWSTrainingModel, self).__init__()
        
        # Variables
        self.label = label
        self.d_model = d_model
        
        # Model used as body
        self.body = body
        
        # Head
        self.head = nn.Linear(3*d_model,label)
    
        # Loss function
        self.loss = nn.CrossEntropyLoss()

    # Define the forward pass
    def forward(self,input_ids1=None,attention_mask1=None,input_ids2=None,attention_mask2=None,labels=None):
        # Get all hidden vectors from the body for the two sentence
        h1 = self.body(input_ids1)
        h2 = self.body(input_ids2)
        
        # Use mean pooling
        u = mean_pooling(h1,attention_mask1)
        v = mean_pooling(h2,attention_mask2)
        
        # Concatenation
        input_concat = []
        input_concat.append(u)
        input_concat.append(v)
        input_concat.append(torch.abs(u-v))
        input_head = torch.cat(input_concat,1)
        
        # Go to the head
        logits = self.head(input_head)
        
        # Compute the loss
        Headloss = self.loss(logits,labels.view(-1))
        
        return Headloss,logits

# Define the model for STSB (Bi-Encoder)
class STSBTrainingModel(nn.Module):
    def __init__(self,body):
        super(STSBTrainingModel, self).__init__()
        
        # Model used as body
        self.body = body
    
        # Loss function
        self.loss = nn.MSELoss()

    def forward(self,input_ids1=None,attention_mask1=None,input_ids2=None,attention_mask2=None,score=None):
        # Get all hidden vectors from the body for the two sentence
        h1 = self.body(input_ids1)
        h2 = self.body(input_ids2)
        
        # Use mean pooling
        u = mean_pooling(h1,attention_mask1)
        v = mean_pooling(h2,attention_mask2)
        
        # Compute cosine similarity
        sim = torch.cosine_similarity(u,v)
        
        # Compute the loss
        Headloss = self.loss(sim,score.view(-1))
        
        return Headloss,sim



if __name__ == "__main__":
    # Inform the user if no GPU is detected
    if torch.cuda.is_available() is True:
        device = "cuda"
    else:
        print("Pas de GPU pour le training, celà sera trop long")
        
    # Set Global Parameters
    max_length = 128
    model_name = "camembert-base"
    model_save1 = "Model_Finetuning/camembert-base1"
    model_save2 = "Model_Finetuning/camembert-base2"
    model_save3 = "Model_Finetuning/camembert-base3"
    batch_size = 16
    warmup = 0.1
    learning_rate = 2e-5
    weight_decay = 0.01
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    num_epochs = 10
    
    # Create the tokenize function
    def tokenize1(df):
        return tokenizer(df["sentence1"],padding=True,truncation=True,max_length=max_length)
    
    def tokenize2(df):
        return tokenizer(df["sentence2"],padding=True,truncation=True,max_length=max_length)
    
    ##
    ## Train model on PAWS-X
    ##  
    
    # # Load the datasets : ['sentence1', 'sentence2', 'label']
    # train,validation,test = getPAWS()
    
    # # Transform in the correct form : ['input_ids1', 'attention_mask1', 'input_ids2', 'attention_mask2','labels']
    # train_encoded = train.map(tokenize1,batched=True,batch_size=None)
    # train_encoded = train_encoded.rename_column("input_ids","input_ids1")
    # train_encoded = train_encoded.rename_column("attention_mask","attention_mask1")
    # train_encoded = train_encoded.map(tokenize2,batched=True,batch_size=None)
    # train_encoded = train_encoded.rename_column("input_ids","input_ids2")
    # train_encoded = train_encoded.rename_column("attention_mask","attention_mask2")
    # train_encoded = train_encoded.rename_column("label","labels")
    # train_encoded = train_encoded.remove_columns(["sentence1"])
    # train_encoded = train_encoded.remove_columns(["sentence2"])
    
    # validation_encoded = validation.map(tokenize1,batched=True,batch_size=None)
    # validation_encoded = validation_encoded.rename_column("input_ids","input_ids1")
    # validation_encoded = validation_encoded.rename_column("attention_mask","attention_mask1")
    # validation_encoded = validation_encoded.map(tokenize2,batched=True,batch_size=None)
    # validation_encoded = validation_encoded.rename_column("input_ids","input_ids2")
    # validation_encoded = validation_encoded.rename_column("attention_mask","attention_mask2")
    # validation_encoded = validation_encoded.rename_column("label","labels")
    # validation_encoded = validation_encoded.remove_columns(["sentence1"])
    # validation_encoded = validation_encoded.remove_columns(["sentence2"])
    
    # test_encoded = test.map(tokenize1,batched=True,batch_size=None)
    # test_encoded = test_encoded.rename_column("input_ids","input_ids1")
    # test_encoded = test_encoded.rename_column("attention_mask","attention_mask1")
    # test_encoded = test_encoded.map(tokenize2,batched=True,batch_size=None)
    # test_encoded = test_encoded.rename_column("input_ids","input_ids2")
    # test_encoded = test_encoded.rename_column("attention_mask","attention_mask2")
    # test_encoded = test_encoded.rename_column("label","labels")
    # test_encoded = test_encoded.remove_columns(["sentence1"])
    # test_encoded = test_encoded.remove_columns(["sentence2"])

    # # Set the correct format
    # train_encoded.set_format("torch")
    # validation_encoded.set_format("torch")
    # test_encoded.set_format("torch")

    # # Create the different Dataloaders
    # trainloader = DataLoader(train_encoded,shuffle=True,batch_size=batch_size)
    # validationloader = DataLoader(validation_encoded,batch_size=batch_size)
    # testloader = DataLoader(test_encoded,batch_size=batch_size)

    # # Load the model used as body
    # body = AutoModel.from_pretrained(model_name,max_length=max_length)

    # # Create the training model
    # model = PAWSTrainingModel(body=body).to(device)
    
    # # Load the model and it
    # optimizer = AdamW(model.parameters(),lr=learning_rate,weight_decay=weight_decay)
    # training_steps = num_epochs*len(trainloader)
    # warmup_steps = math.ceil(training_steps*warmup)
    # scheduler = get_constant_schedule_with_warmup(optimizer=optimizer,num_warmup_steps=warmup_steps)

    # # Set up the progress bar
    # progress_bar = tqdm(range(training_steps))

    # # Number of steps counter
    # count = 0
    
    # # Loss keeper
    # loss_keep = []

    # # Train the model
    # for epoch in range(num_epochs):
    #     model.train()
    #     for batch in trainloader:
    #         # Clear the gradient
    #         optimizer.zero_grad()
    #         # Batch to GPU
    #         batch = {k: v.to(device) for k, v in batch.items()}
    #         # Predict the batch
    #         loss,_ = model(**batch)
    #         # Compute the gradient
    #         loss.backward()
    #         # Make the step of training
    #         optimizer.step()
    #         scheduler.step()
    #         # Update the progess bar
    #         progress_bar.update(1)
    #         # Update the count
    #         count = count + 1
        
    #         if count == 200:
    #             model.eval()
    #             count = 0
    #             tmp_loss = []
    #             for batch in validationloader:
    #                 # Batch to GPU
    #                 batch = {k: v.to(device) for k, v in batch.items()}
    #                 # Predict the batch (no gradients needed)
    #                 with torch.no_grad():
    #                     loss,_ = model(**batch)
    #                 # Append the loss
    #                 tmp_loss.append(loss.item())
    #             # Make the loss independant to the batch size
    #             tmp_loss = sum(tmp_loss)/len(validationloader)
    #             # Save the loss
    #             loss_keep.append(tmp_loss)
    #             # Reset model train
    #             model.train()

    # # Save the trained model with the tokenizer
    # trainer = Trainer(model=body,tokenizer=tokenizer)
    # trainer.save_model(model_save1)


    ##
    ## Train model on XNLI
    ##
    
    # # Load the datasets
    # train,validation,test = getXNLI()
    
    # # Transform in the correct form : ['input_ids1', 'attention_mask1', 'input_ids2', 'attention_mask2','labels']
    # train_encoded = train.map(tokenize1,batched=True,batch_size=None)
    # train_encoded = train_encoded.rename_column("input_ids","input_ids1")
    # train_encoded = train_encoded.rename_column("attention_mask","attention_mask1")
    # train_encoded = train_encoded.map(tokenize2,batched=True,batch_size=None)
    # train_encoded = train_encoded.rename_column("input_ids","input_ids2")
    # train_encoded = train_encoded.rename_column("attention_mask","attention_mask2")
    # train_encoded = train_encoded.rename_column("label","labels")
    # train_encoded = train_encoded.remove_columns(["sentence1"])
    # train_encoded = train_encoded.remove_columns(["sentence2"])
    
    # validation_encoded = validation.map(tokenize1,batched=True,batch_size=None)
    # validation_encoded = validation_encoded.rename_column("input_ids","input_ids1")
    # validation_encoded = validation_encoded.rename_column("attention_mask","attention_mask1")
    # validation_encoded = validation_encoded.map(tokenize2,batched=True,batch_size=None)
    # validation_encoded = validation_encoded.rename_column("input_ids","input_ids2")
    # validation_encoded = validation_encoded.rename_column("attention_mask","attention_mask2")
    # validation_encoded = validation_encoded.rename_column("label","labels")
    # validation_encoded = validation_encoded.remove_columns(["sentence1"])
    # validation_encoded = validation_encoded.remove_columns(["sentence2"])
    
    # test_encoded = test.map(tokenize1,batched=True,batch_size=None)
    # test_encoded = test_encoded.rename_column("input_ids","input_ids1")
    # test_encoded = test_encoded.rename_column("attention_mask","attention_mask1")
    # test_encoded = test_encoded.map(tokenize2,batched=True,batch_size=None)
    # test_encoded = test_encoded.rename_column("input_ids","input_ids2")
    # test_encoded = test_encoded.rename_column("attention_mask","attention_mask2")
    # test_encoded = test_encoded.rename_column("label","labels")
    # test_encoded = test_encoded.remove_columns(["sentence1"])
    # test_encoded = test_encoded.remove_columns(["sentence2"])
    
    # # Set the correct format
    # train_encoded.set_format("torch")
    # validation_encoded.set_format("torch")
    # test_encoded.set_format("torch")
    
    # # Create the different Dataloaders
    # trainloader = DataLoader(train_encoded,shuffle=True,batch_size=batch_size)
    # validationloader = DataLoader(validation_encoded,batch_size=batch_size)
    # testloader = DataLoader(test_encoded,batch_size=batch_size)

    # # Load the model used as body
    # body = AutoModel.from_pretrained(model_save1,max_length=max_length)

    # # Create the training model
    # model = XNLITrainingModel(body=body).to(device)
    
    # # Load the model and it
    # optimizer = AdamW(model.parameters(),lr=learning_rate,weight_decay=weight_decay)
    # training_steps = num_epochs*len(trainloader)
    # warmup_steps = math.ceil(training_steps*warmup)
    # scheduler = get_constant_schedule_with_warmup(optimizer=optimizer,num_warmup_steps=warmup_steps)

    # # Set up the progress bar
    # progress_bar = tqdm(range(training_steps))

    # # Number of steps counter
    # count = 0
    
    # # Loss keeper
    # loss_keep = []

    # # Train the model
    # for epoch in range(num_epochs):
    #     model.train()
    #     for batch in trainloader:
    #         # Clear the gradient
    #         optimizer.zero_grad()
    #         # Batch to GPU
    #         batch = {k: v.to(device) for k, v in batch.items()}
    #         # Predict the batch
    #         loss,_ = model(**batch)
    #         # Compute the gradient
    #         loss.backward()
    #         # Make the step of training
    #         optimizer.step()
    #         scheduler.step()
    #         # Update the progess bar
    #         progress_bar.update(1)
    #         # Update the count
    #         count = count + 1
        
    #         if count == 200:
    #             model.eval()
    #             count = 0
    #             tmp_loss = []
    #             for batch in validationloader:
    #                 # Batch to GPU
    #                 batch = {k: v.to(device) for k, v in batch.items()}
    #                 # Predict the batch (no gradients needed)
    #                 with torch.no_grad():
    #                     loss,_ = model(**batch)
    #                 # Append the loss
    #                 tmp_loss.append(loss.item())
    #             # Make the loss independant to the batch size
    #             tmp_loss = sum(tmp_loss)/len(validationloader)
    #             # Save the loss
    #             loss_keep.append(tmp_loss)
    #             # Reset model train
    #             model.train()

    # # Save the trained model with the tokenizer
    # trainer = Trainer(model=body,tokenizer=tokenizer)
    # trainer.save_model(model_save2)

    
    ##
    ## Train model on STSB
    ##
    
    # Load the datasets
    # rescaleSTSB() # Need to be done at least one time
    train,validation,test = getSTSB()

    # Transform in the correct form : ['input_ids1', 'attention_mask1', 'input_ids2', 'attention_mask2','score']
    train_encoded = train.map(tokenize1,batched=True,batch_size=None)
    train_encoded = train_encoded.rename_column("input_ids","input_ids1")
    train_encoded = train_encoded.rename_column("attention_mask","attention_mask1")
    train_encoded = train_encoded.map(tokenize2,batched=True,batch_size=None)
    train_encoded = train_encoded.rename_column("input_ids","input_ids2")
    train_encoded = train_encoded.rename_column("attention_mask","attention_mask2")
    train_encoded = train_encoded.remove_columns(["sentence1"])
    train_encoded = train_encoded.remove_columns(["sentence2"])
    
    validation_encoded = validation.map(tokenize1,batched=True,batch_size=None)
    validation_encoded = validation_encoded.rename_column("input_ids","input_ids1")
    validation_encoded = validation_encoded.rename_column("attention_mask","attention_mask1")
    validation_encoded = validation_encoded.map(tokenize2,batched=True,batch_size=None)
    validation_encoded = validation_encoded.rename_column("input_ids","input_ids2")
    validation_encoded = validation_encoded.rename_column("attention_mask","attention_mask2")
    validation_encoded = validation_encoded.remove_columns(["sentence1"])
    validation_encoded = validation_encoded.remove_columns(["sentence2"])
    
    test_encoded = test.map(tokenize1,batched=True,batch_size=None)
    test_encoded = test_encoded.rename_column("input_ids","input_ids1")
    test_encoded = test_encoded.rename_column("attention_mask","attention_mask1")
    test_encoded = test_encoded.map(tokenize2,batched=True,batch_size=None)
    test_encoded = test_encoded.rename_column("input_ids","input_ids2")
    test_encoded = test_encoded.rename_column("attention_mask","attention_mask2")
    test_encoded = test_encoded.remove_columns(["sentence1"])
    test_encoded = test_encoded.remove_columns(["sentence2"])
    
    # Set the correct format
    train_encoded.set_format("torch")
    validation_encoded.set_format("torch")
    test_encoded.set_format("torch")
    
    # Create the different Dataloaders
    trainloader = DataLoader(train_encoded,shuffle=True,batch_size=batch_size)
    validationloader = DataLoader(validation_encoded,batch_size=batch_size)
    testloader = DataLoader(test_encoded,batch_size=batch_size)

    # Load the model used as body
    body = AutoModel.from_pretrained(model_name,max_length=max_length)

    # Create the training model
    model = STSBTrainingModel(body=body).to(device)
    
    # Load the model and it
    optimizer = AdamW(model.parameters(),lr=learning_rate,weight_decay=weight_decay)
    training_steps = num_epochs*len(trainloader)
    warmup_steps = math.ceil(training_steps*warmup)
    scheduler = get_constant_schedule_with_warmup(optimizer=optimizer,num_warmup_steps=warmup_steps)

    # Set up the progress bar
    progress_bar = tqdm(range(training_steps))

    # Loss keeper
    loss_keep = []
    loss_values_train = []
    
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
    loss_values_train.append(tmp_loss)
    
    tmp_loss = []
    for batch in validationloader:
        # Batch to GPU
        batch = {k: v.to(device) for k, v in batch.items()}
        # Predict the batch (no gradients needed)
        with torch.no_grad():
            loss,_ = model(**batch)
        # Append the loss
        tmp_loss.append(loss.item())
    # Make the loss independant to the batch size
    tmp_loss = sum(tmp_loss)/len(validationloader)
    # Save the loss
    loss_keep.append(tmp_loss)

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
        loss_values_train.append(tmp_loss)
        
        model.eval()
        tmp_loss = []
        for batch in validationloader:
            # Batch to GPU
            batch = {k: v.to(device) for k, v in batch.items()}
            # Predict the batch (no gradients needed)
            with torch.no_grad():
                loss,_ = model(**batch)
            # Append the loss
            tmp_loss.append(loss.item())
        # Make the loss independant to the batch size
        tmp_loss = sum(tmp_loss)/len(validationloader)
        # Save the loss
        loss_keep.append(tmp_loss)

    # Plot the loss curves
    plt.plot(loss_values_train,"r",label="train set")
    plt.plot(loss_keep,"b",label = "validation set")
    plt.title("Training of model Base for Intents Classification")
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # Save the trained model with the tokenizer
    trainer = Trainer(model=body,tokenizer=tokenizer)
    trainer.save_model(model_save3)

    
    
    