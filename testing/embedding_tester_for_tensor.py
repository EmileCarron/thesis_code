from bs4 import BeautifulSoup
import pandas as pd
import torch
import brambox as bb
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

tensor = torch.load('/Users/emilecarron/test/tankstation_results/retinanet_embedding_rp2k/tensor_p1.pt')
#print(tensor)

embedding = []
labels = []
for x in range(len(tensor[0]['boxes'])):
    if tensor[0]['scores'][x].item() > 0.2:
        embedding.append(tensor[0]['embedding'][x])
        labels.append(tensor[0]['labels'][x].item())
        
        
col_names = []
embed= []
for x in range(len(embedding[0])):
    name = x
    col_names.append(str(name))

for x in range(len(embedding)):
    emb = embedding[x].detach().numpy()
    embed.append(emb)
    
list_emb = tuple([tuple(e) for e in embed])



df_emb = pd.DataFrame(list_emb, columns =col_names)

print(df_emb)

match = 0 - len(embed)
counter = 0 - len(embed)
mustmatch = 0 - len(embed)
wrongmatch =  0
counter = 0
for e in range(len(embed)):
    for y in range(len(embed)-e):
        if labels[e] == labels[e+y]:
            mustmatch = mustmatch + 1

for e in range(len(embed)):
    for y in range(len(embed)):
        res = np.linalg.norm(embed[e]-embed[y])
        counter = counter + 1
        if res < 1 and labels[e] != labels[y]:
            wrongmatch = wrongmatch + 1
        if res < 1 and labels[e] == labels[y]:
            match = match + 1
            
print('Must match: ', mustmatch)
print('Match: ', match)
print('Wrong match', wrongmatch)

import pdb; pdb.set_trace()
print('end')

