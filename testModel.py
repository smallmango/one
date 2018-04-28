import torch
import torch.utils.data
from torch.autograd import Variable
import numpy as np


def test(dataLoader, model, lossFun):
    model.eval()
    totalLoss = 0.0
    totalMAE = 0.0
    totalNum = 0.0
    for step, (img, age, ageBinary) in enumerate(dataLoader):
        if torch.cuda.is_available():
            x = Variable(img, volatile = True).type(torch.cuda.FloatTensor)
            y = Variable(ageBinary, volatile = True).type(torch.cuda.FloatTensor)
        else:
            x = Variable(img, volatile = True).float()
            y = Variable(ageBinary, volatile = True).float()
        output = model(x)
        loss = lossFun(output, y)
        pred = (output > 0.5).data.cpu().numpy().sum(axis=1) + 2
        MAE = np.abs(pred - age.numpy()).mean()

        totalLoss += loss.data[0]
        totalMAE += MAE
        totalNum += 1
    return ((totalLoss/totalNum), (totalMAE/totalNum))


