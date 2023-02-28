"""
Example of how to use the MetadataDistillationModel class to train a model to distill knowledge from metadata to text.

"""
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import pickle
import nltk
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from model import MetadataDistillationModel 
from utils.pre_processing import semicolons, removeSectionsNumber, removeParenthesis, removeUnicode, removeWeirdChars, remove_html_tags, replaceRoman

#read xlsx and convert to csv
df = pd.read_excel(r"C:\Users\Rui\Documents\GitHub\metadata-knowledge-distillation\data.xlsx")


#Preprocessing data
df_copy = df.copy()[0:2]
df_copy["sumario"] = df_copy["sumario"].str.replace("\t","")
df_copy["sumario"] = df_copy["sumario"].str.replace("\n","")
df_copy.drop_duplicates(subset ="sumario", keep = False, inplace = True)
df_copy.drop(columns=["data"], inplace=True)

#Specific functions for out dataset
def get_all_descritores(df_copy):
    descritores = []
    #convert descritores to list of strings
    df_copy['descritores_list'] = df_copy['descritores'].progress_apply(lambda x: eval(x))
    for index, row in df_copy.iterrows():
        descritores.extend(row['descritores_list'])

    return set(descritores)
def find_entries_index_with_descritor(descritor, df):
    return df[df["descritores"].str.contains(descritor, na=False)].index.tolist()


#Load model
legalBertimbau = MetadataDistillationModel('rufimelo/Legal-BERTimbau-large-TSDAE-v4-GPL-sts')

#Create dataframe with sentences and embeddings
df_sentences = pd.DataFrame(columns=['descritores','sentences', 'embedding'])
for index, row in tqdm(df_copy.iterrows(), total=df_copy.shape[0]):
    summary = row['sumario']
    summary = semicolons(removeSectionsNumber(removeParenthesis(removeUnicode(removeWeirdChars(remove_html_tags(replaceRoman(summary))))))).strip()
    
    descritor = row['descritores']
    #get sentences using nltk
    sentences = nltk.sent_tokenize(summary)
    sentences = [sentence for sentence in sentences if len(sentence)>50]
    
    #add to dataframe
    for sentence in sentences:
        df_sentences = df_sentences.append({'descritores':descritor, 'sentences': sentence, 'embedding': legalBertimbau.encode_original(sentence)}, ignore_index=True)
    

#adjust embeddings to be closer to the centroid of the embeddings of the same descritor
descritores = list(get_all_descritores(df_sentences))
for descritor in descritores:
    indexes = find_entries_index_with_descritor(descritor, df_sentences)

    #find centroid of the embeddings
    centroid = np.mean(df_sentences.iloc[indexes]['embedding'].tolist(), axis=0)
    
    #adjust the embeddings 1% closer to the centroid
    df_sentences.iloc[indexes]['embedding'] = df_sentences.iloc[indexes]['embedding'].progress_apply(lambda x: x + (centroid - x)*0.01)


#save dataframe
with open('df_sentences.pickle', 'wb') as handle:
    pickle.dump(df_sentences, handle, protocol=pickle.HIGHEST_PROTOCOL)

#with open('df_sentences.pickle', 'rb') as handle:
#    df_sentences = pickle.load(handle)


#train model
X = df_sentences['sentences']
y = torch.tensor(df_sentences['embedding'].tolist())
#associate to cuda
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
legalBertimbau.to(device)

legalBertimbau.train(X, torch.tensor(y), 1, lr=1e-6, batch_size=3)

#save model
legalBertimbau.save('exampleModel')
