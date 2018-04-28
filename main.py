import torch
import torch.utils.data as Data
import Data_Process
import Model
from torch import nn
from torch.autograd import Variable
import testModel
import numpy as np
import trainModel
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

torch.manual_seed(2)

BATCH_SIZE = 64
EPOCH = 50
LR = 0.01
WEIGHT_DECAY = 1e-5
TEST_SETP = 5


if __name__ == '__main__':

    trainData, testData = Data_Process.dataProcess()

    trainLoader = Data.DataLoader(
        dataset=trainData,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
    )

    testLoader = Data.DataLoader(
        dataset=testData,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
    )

    if torch.cuda.is_available():
        model = Model.Net().cuda()
    else:
        model = Model.Net()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    lossfun = nn.BCELoss().cuda()

    trainLoss = np.zeros([EPOCH, 2]) # save ageLoss and MAE
    testLoss = np.zeros([EPOCH, 2])

    for epoch in range(EPOCH):
        ageLoss, MAE = trainModel.train(trainLoader, model, lossfun, optimizer)
        trainLoss[epoch, :] = [ageLoss, MAE]
        print('Training | Epoch : %d  |  loss = %.4f  |  MAE = %.2f' % ((epoch + 1), ageLoss, MAE))
        if (epoch + 1) % TEST_SETP == 0:
            ageLoss, MAE = testModel.test(testLoader, model, lossfun)
            testLoss[epoch, :] = [ageLoss, MAE]
            print('-----------------------TEST-----------------------')
            print('Testing | Epoch : %d  |  loss = %.4f  |  MAE = %.2f' % ((epoch + 1), ageLoss, MAE))
            print('--------------------------------------------------')

