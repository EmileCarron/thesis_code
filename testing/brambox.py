from bs4 import BeautifulSoup
import pandas as pd
import torch
import brambox as bb
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple, Optional
from torch import nn, Tensor
import torchvision
from torchvision import transforms
import os

# This code test the IoU and provides the labels from the testset to our detections so we can compare the labels with the embeddings

list_of_path = []
list_of_image = []
path =r'/Users/emilecarron/test/tankstation_detection/'
for root, directories, file in os.walk(path):
    for file in file:
        if(file.endswith(".jpg")):
            list_of_path.append(os.path.join(root,file))
            file = file.split('.')
            list_of_image.append(file[0])


for y in range(len(list_of_image)):
    # Reading the data inside the xml file to a variable under the name  data
    with open('tankstation_detection/'+ list_of_image[y] + '.xml', 'r') as f:
        data = f.read()

    # Passing the stored data inside the beautifulsoup parser
    bs_data = BeautifulSoup(data, 'xml')

    # Finding all instances of tag
    b_unique_name = bs_data.find_all('name')
    b_unique_name = list(dict.fromkeys(b_unique_name))
    b_unique = bs_data.find_all('object')

    xmin = []
    ymin = []
    width = []
    height = []
    label = []
    for x in range(len(b_unique)):
        xmin.append(b_unique[x].xmin.get_text())
        ymin.append(b_unique[x].ymin.get_text())
        height.append(int(b_unique[x].ymax.get_text())-int(b_unique[x].ymin.get_text()))
        width.append(int(b_unique[x].xmax.get_text())-int(b_unique[x].xmin.get_text()))
        for name in range(len(b_unique_name)):
            if b_unique[x].find_all('name')[0] == b_unique_name[name]:
                label.append(name)

    d = {'image':bs_data.filename, 'class_label':label,'id': [0]*len(b_unique), 'x_top_left': xmin, 'y_top_left': ymin, 'width': width, 'height': height, 'occluded':[1]*len(b_unique), 'ignore': [False]*len(b_unique)}
    d2 = {'image':bs_data.filename, 'class_label':[1]*len(b_unique),'id': [0]*len(b_unique), 'x_top_left': xmin, 'y_top_left': ymin, 'width': width, 'height': height, 'occluded':[1]*len(b_unique), 'ignore': [False]*len(b_unique)}

    label = pd.DataFrame(data=d)
    anno = pd.DataFrame(data=d2)

    anno['x_top_left'] = anno['x_top_left'].astype(float, errors = 'raise')
    anno['y_top_left'] = anno['y_top_left'].astype(float, errors = 'raise')
    anno['width'] = anno['width'].astype(float, errors = 'raise')
    anno['height'] = anno['height'].astype(float, errors = 'raise')

    tensor2 = torch.load('tankstation_results/retinanet_embedding_aliprod/tensor_'+ list_of_image[y] + '.pt')
    tensor = tensor2

    xmin = []
    ymin = []
    width = []
    height = []
    confidence = []
    embedding = []
    labels = []
    for x in range(len(tensor[0]['boxes'])):
        if tensor[0]['scores'][x].item() > 0.2:
            xmin.append(tensor[0]['boxes'][x][0].item())
            ymin.append(tensor[0]['boxes'][x][1].item())
            width.append(tensor[0]['boxes'][x][2].item()-tensor[0]['boxes'][x][0].item())
            height.append(tensor[0]['boxes'][x][3].item()-tensor[0]['boxes'][x][1].item())
            confidence.append(tensor[0]['scores'][x].item())
            embedding.append(tensor[0]['embedding'][x])
            #labels.append(tensor[0]['labels'][x].item())
            labels.append(tensor[0]['labels'][x])

    d = {'image':bs_data.filename, 'class_label':[1]*len(confidence),'id': [0]*len(confidence), 'x_top_left': xmin, 'y_top_left': ymin, 'width': width, 'height': height, 'confidence':confidence}
    det = pd.DataFrame(data=d)
 

    matched_det = bb.stat.match_box(
        det, anno,
        threshold=0.5,                       # Matching threshold (Here IoU, see matching function)
        criteria=bb.stat.coordinates.iou,    # Matching function
        #ignore=bb.stat.IgnoreMethod.SINGLE,  # Only allow to match with ignored anno once
    )
    print(matched_det)
    print(matched_det[1]['detection'])
    pr = bb.stat.pr(matched_det[0], anno)
    ap = bb.stat.ap(pr)

    detections: List[Dict[str, Tensor]] = []
    good_tensor = matched_det[1]

    good_tensor = good_tensor.fillna(0)
    matched_det[1].to_csv("tankstation_results/retinanet_embedding_aliprod/matched_det_"+ list_of_image[y] + ".csv")

    lab = []
    for l in range(len(label.class_label)):
        lab.append(label.class_label[l].item())

    tenbox = tensor2[0]['boxes'].detach().numpy()
    tenscore = tensor2[0]['scores'].detach().numpy()
    tenembedding = tensor2[0]['embedding'].detach().numpy()
    boxes = []
    scores= []
    labels = []
    embedding = []
    
    for z in range(len(good_tensor)):
        if good_tensor['detection'][z] != 0:
                    boxes.append(tenbox[int(good_tensor['detection'][z])])
                    scores.append(tenscore[int(good_tensor['detection'][z])])
                    labels.append(label['class_label'][z])
                    embedding.append(tenembedding[int(good_tensor['detection'][z])])
                    
    detections.append({
                'boxes': boxes,
                'scores': scores,
                'labels': labels,
                'embedding': embedding,
                })
                 
    #print(detections)
    torch.save(detections, 'tankstation_results/retinanet_embedding_aliprod/labeled'+ list_of_image[y] + '.pt')

