import torch
import torch.nn as nn
import torch.nn.functional as F


class Sigmoid(nn.Module):
    def forward(self, input,a,b):
        x = 1/(1 + a*torch.exp(-input*b))
        return x
sigmoid = Sigmoid()

def predict(data, label, model):
    corr = 0
    acc = 0
    for i in range(data.shape[0]): 
        x = data[i].unsqueeze_(0)
        y = label[i].unsqueeze_(0)
        pred = model(x)
        prediction = pred.data.max(1, keepdim=True)[1][0].item()
        if prediction==y:
            corr+=1
        #print("Accuracy: %.2f" % (float(corr/(i+1)*100)), i)
    acc = float(corr/data.shape[0]*100)
    #print("Overall Accuracy: %.2f" % (float(corr/data.shape[0]*100)))
    return acc


    
