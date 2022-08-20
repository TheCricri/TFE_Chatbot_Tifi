"""
TFE - Chatbot Tifi - Technifutur
by Nicolas Christiaens
"""

import pandas as pd
import html
import json
import urllib.request
import re

# Function to get all the formations from technifutur
def getAllFormationsFromWEB():
    URL = "http://private.technifutur.be/wbs/formations.asp"
    df = pd.read_json(URL)
    df = pd.json_normalize(df["formations"])
    return df

# Function to save the database into a json local file
def saveDFtoJSON(df):
    df.to_json("DB")

# Function to read the json local file into a database
def readDF():
    try:
        df = pd.read_json("DB")
    except:
        df = None
        print("No Database save in a json file")
        
    return df
    
# Function to clear a html text column
def HTMLDecoder(col,tags=True):
    # Convert the text in unicode characters
    col = col.apply(lambda x: html.unescape(x))
    
    # Remove the tags
    if tags is True:
        col = col.apply(lambda x: removeTags(x))
    return col

# Function to get detailed information of a specific formation
def getInformation(row):
    idax = row["idax"]
    URL = "http://private.technifutur.be/wbs/formation.asp?id=" + idax
    try:
        link = urllib.request.urlopen(URL)
        info = json.loads(link.read().decode())
    except:
        return None
    return info

# Function to get the detailed informations of all the formations
def getInformationsForAll(df):
    df["info"] = df.apply(lambda row: getInformation(row),axis=1)
    return df

# Function to handle all the process needed for creating the database
def processAll(save=True):
    # Create the database and extract information while eliminating duplicates and 
    # formations whose detailed information is not available
    df = getAllFormationsFromWEB()
    df = df.drop_duplicates(subset=['intitule'])
    df = getInformationsForAll(df)
    df2 = pd.json_normalize(df["info"])
    df2 = df2[df2['intitule'].notna()]
    df = df.drop("info",axis=1)
    df = df.merge(df2)
    
    # Clear the database of hmtl text and tags
    df.intitule = HTMLDecoder(df.intitule,tags=False)
    df.duree_txt = HTMLDecoder(df.duree_txt,tags=False)
    df.methodo = HTMLDecoder(df.methodo)
    df.programme = HTMLDecoder(df.programme)
    df.prerequis = HTMLDecoder(df.prerequis)
    df.objectif = HTMLDecoder(df.objectif)
    df.cible = HTMLDecoder(df.cible)
    
    # Save if needed
    if save is True:
        saveDFtoJSON(df)
    return df

# Remove specific tags or space based on string patterns
def removeTags(text):
    # Tags changing
    start_list2 = re.compile(r"</h[0-9]><p.*?>(.*?)</p><ul><li>")
    text = start_list2.sub(r"\1:",text)
    start_list = re.compile(r"</h[0-9]><ul><li>")
    text = start_list.sub(": ",text)
    tag_list = re.compile(r"</li><li>")
    text = tag_list.sub(", ",text)
    end_list2 = re.compile(r"</li></ul><p>")
    text = end_list2.sub(")\n",text)
    end_list3 = re.compile(r"</li></ul></ul>")
    text = end_list3.sub(")\n",text)
    end_list4 = re.compile(r"</li></ul><li>")
    text = end_list4.sub("), ",text)
    end_list = re.compile(r"</ul><.*?>")
    text = end_list.sub("\n",text)
    end_div = re.compile(r"</div><ul><li>")
    text = end_div.sub("\n",text)
    end_saviez = re.compile(r"</a></p></div><p>")
    text = end_saviez.sub("\n",text)
    sub_open = re.compile(r":<ul><li>")
    text = sub_open.sub(" (",text)
    sub_open2 = re.compile(r"</p><ul><li>")
    text = sub_open2.sub(" (",text)
    sub_open3 = re.compile(r"</li><ul><li>")
    text = sub_open3.sub(" (",text)
    sub_end = re.compile(r"</li></ul>")
    text = sub_end.sub(") ",text)
    gras = re.compile(r"</p><p>")
    text = gras.sub("\n",text)
    sub_allh = re.compile(r"</h[0-9]><h[0-9]>")
    text = sub_allh.sub("\n",text)
    main_title = re.compile(r"<h[0-9].*?>")
    text = main_title.sub("\n",text)
    wtf1 = re.compile(r"</strong></span>")
    text = wtf1.sub(": ",text)
    tag = re.compile(r"<.*?>")
    text = tag.sub(" ",text)

    # Cosmetic change
    text = text.replace(u"\xa0"," ")
    text = re.sub(r" , ",", ",text)
    text = re.sub(r" +"," ",text)
    text = re.sub(r"\n\n","\n",text)
    text = re.sub(r"::",":",text)
    text = re.sub(r",,",",",text)
    text = re.sub(r";,",",",text)
    while text != "":
        if text[0] == " ":
            text = text[1:]
        elif text[-1] == " ":
            text = text[:-1]
        else:
            break
    if text != "":
        count1 = text.count("(")
        count2 = text.count(")")
        if (count1 < count2) and (text[-1] == ")"):
            text = text[:-1]
    text = re.sub(r"\n ","\n",text)
    text = text.replace(r".,",",")
    text = text.replace(r"..,","...,")
    return text

# Function to extract contacts from the database (Warning : use df.iloc[[index]] for "row" input)
def extractContacts(row,dom):
    if (dom != "ens") & (dom != "ent") & (dom != "de"):
        print("Bad contact domain")
        return None
    col = "contact_" + dom
    tmp = row[col].tolist()
    if tmp[0] is None:
        return None
    df = pd.json_normalize(tmp[0])
    df = df.apply(lambda col: HTMLDecoder(col,tags=False),axis=0)
    return df

# Function to extract planning from the database (Warning : use df.iloc[[index]] for "row" input)
def extractPlanning(row):
    col = "planning"
    tmp = row[col].tolist()
    if tmp[0] is None:
        return None
    df = pd.json_normalize(tmp[0])
    return df

if __name__ == "__main__":
    # processAll()
    df = readDF()