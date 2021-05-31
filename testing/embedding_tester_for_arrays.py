from bs4 import BeautifulSoup
import pandas as pd
import torch
import brambox as bb
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import os

list_of_path = []
list_of_image = []
path =r'/Users/emilecarron/test/tankstation_detection/'
for root, directories, file in os.walk(path):
    for file in file:
        if(file.endswith(".jpg")):
            list_of_path.append(os.path.join(root,file))
            file = file.split('.')
            list_of_image.append(file[0])



restupple = []




            
for l in range(len(list_of_image)):

    tensor = torch.load('/Users/emilecarron/test/tankstation_results/retinanet_embedding_rp2k/labeled' + list_of_image[l] + '.pt')
    #print(tensor)
    #import pdb; pdb.set_trace()
    embedding = tensor[0]['embedding']
    labels = tensor[0]['labels']

            
            
    col_names = []
    embed= []
    for x in range(len(embedding[0])):
        name = x
        col_names.append(str(name))

    for x in range(len(embedding)):
        emb = embedding[x]
        embed.append(emb)
        
    list_emb = tuple([tuple(e) for e in embed])



    df_emb = pd.DataFrame(list_emb, columns =col_names)

    #print(df_emb)



    similarity = cosine_similarity(df_emb)
    #print(similarity[0][1])
    #for x in range(len(embedding)):
    #print(similarity[1])
    match = similarity[1]>0.999
    #print(match)
    result = np.where(match == True)
    #print(result)
    match = 0
    counter = 0
    mustmatch = 0
    wrongmatch =  0


    for e in range(len(embed)):
        for y in range(len(embed)-e-1):
            if labels[e] == labels[e+y+1]:
                mustmatch = mustmatch + 1

    for e in range(len(embed)):
        for y in range(len(embed)-e-1):
            res = np.linalg.norm(embed[e]-embed[e+y+1])
            counter = counter + 1
            if res < 1 and labels[e] != labels[e+y+1]:
                wrongmatch = wrongmatch + 1
            if res < 1 and labels[e] == labels[e+y+1]:
                match = match + 1
                
                    
    #import pdb; pdb.set_trace()
    
    restupple.append([list_of_image[l],mustmatch,match,wrongmatch,mustmatch-match])

results = pd.DataFrame(restupple, columns =["Image name", "All positives", "True Positives", "False Positives", "False negatives"])

results.to_csv("tankstation_results/retinanet_embedding_rp2k/results_embedding.csv")

