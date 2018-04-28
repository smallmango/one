import torch.utils.data as data
import numpy as np
import os
from PIL import Image
from torchvision import transforms
import csv
import torch



def readTxt(path):
    indexList = []
    with open(path, 'r') as f:
        data = f.readlines()
        for line in data:
            indexList.append(line.strip('\n'))

    return indexList



def dataProcess():
    trainListPath = '/home/ghli/data/OULP-Age/GEI_IDList_train.txt'
    trainList = readTxt(trainListPath)
    dataset = {}
    dataset['name'] = 'GEI'
    dataset['path'] = '/home/ghli/data/OULP-Age/GEI'
    dataset['labelPath'] = '/home/ghli/data/OULP-Age/Age_gender_info.csv'
    dataset['type'] = 'train'
    dataset['indexList'] = trainList
    trainDataset = ImageFolder(dataset)
    #print(trainDataset[0])
    testListPath = '/home/ghli/data/OULP-Age/GEI_IDList_test.txt'
    testList = readTxt(testListPath)
    dataset['type'] = 'test'
    dataset['indexList'] = testList
    testDataset = ImageFolder(dataset)
    return trainDataset, testDataset





class ImageFolder(data.Dataset):
    def __init__(self, dataset):
        self.name = dataset['name']
        self.path = dataset['path']
        self.type = dataset['type']
        self.labelPath = dataset['labelPath']
        self.indexList = dataset['indexList']

        self.imgList = self.loadImg()
        self.ageList, self.genderList = self.loadLabel()
        self.lenght = len(self.imgList)

    def get_processor(self):
        proc = [transforms.ToTensor(), ]
        return transforms.Compose(proc)

    def loadImg(self):
        imgList = []
        processer = self.get_processor()
        for i, number in enumerate(self.indexList):
            path = os.path.join(self.path, number + '.png')
            img = Image.open(path)
            img = img.convert('L')
            #img = self.remove_mean(img)
            imgList.append(processer(img))
        return imgList

    def loadLabel(self):
        ageList , genderList = self.readCsv()
        return ageList, genderList

    def readCsv(self):
        dict = {}
        ageList = []
        genderList = []

        with open(self.labelPath) as f:
            reader = csv.reader(f)
            head_row = next(reader)
            for row in reader:
                id = int(row[0])
                age = row[1]
                gender = row[2]
                dict[id] = {'age': age, 'gender': gender}
        for i, number in enumerate(self.indexList):
            ageList.append(int(dict[int(number)]['age']))
            if(dict[int(number)]['gender'] == 'M'): # M -> 1, F -> 0
                genderList.append(1)
            else:
                genderList.append(0)

        return ageList, genderList


    def __getitem__(self, item):
        img = self.imgList[item]
        age = self.ageList[item]
        gender = self.genderList[item]
        ageBinary = np.zeros(89).astype(int) # 1 stand for gender
        ageBinary[:int(age)] = 1
        # ageBinary[-1] = gender
        #ageBinary = torch.LongTensor(ageBinary)
        return (img, age, ageBinary)

    def __len__(self):
        return self.lenght

    # remove the influence of light
    def remove_mean(self, img):
        img = np.array(img)
        w, h, ch = img.shape
        img = img.reshape(-1, ch)
        m_val = img.mean(axis=0).reshape(1, -1)
        m_val = np.repeat(m_val, w * h, axis=0)
        img = img - m_val
        out = img.reshape(w, h, ch)
        return out


if __name__ == '__main__':

    trainData, testData = dataProcess()

