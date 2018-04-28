import torch
from torch.autograd import Variable
import numpy as np


def train(dataLoader, model, lossFun, optimizer):
    model.train()
    totalLoss = 0.0
    totalMAE = 0.0
    totalNum = 0.0
    for step, (img, age, ageBinary) in enumerate(dataLoader):
        if torch.cuda.is_available():
            x = Variable(img).type(torch.cuda.FloatTensor)
            y = Variable(ageBinary).type(torch.cuda.FloatTensor)
        else:
            x = Variable(img).float()
            y = Variable(ageBinary).float()
        output = model(x)
        loss = lossFun(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pred = (output > 0.5).data.cpu().numpy().sum(axis=1) + 2 # 2 ~ 90
        MAE = np.abs(pred - age.numpy()).mean()

        totalLoss += loss.data[0]
        totalMAE += MAE
        totalNum += 1
    return ((totalLoss/totalNum), (totalMAE/totalNum))




