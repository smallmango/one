import torch.nn as nn
import torch



class Net(nn.Module):
    def __init__(self, input_channels = 1, output_channels = 32, classes = 89):
        super(Net, self).__init__()
        self.single = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 7, stride= 1, padding= 1), # 1, 128 * 88
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(output_channels, output_channels * 2, 5, stride= 1, padding= 0),  # 32, 62 * 42
            nn.BatchNorm2d(output_channels * 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(output_channels * 2, output_channels * 4, 5, stride= 1, padding= 0), # 64, 29 * 19
            nn.BatchNorm2d(output_channels * 4),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 128, 12 * 7
        )
        self.single_fc = nn.Sequential(
            nn.Linear(128 * 12 * 7, classes),
            nn.Sigmoid()
        )
        '''
        self.small_scale = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 5, stride= 2, padding= 0), # 32, 42 * 62
            nn.ReLU(),
            nn.Conv2d(output_channels, output_channels * 2, 5, stride= 2, padding= 0), # 64, 19 * 29
            nn.ReLU(),
            nn.Conv2d(output_channels * 2, output_channels * 4, 3, stride= 1, padding= 0), # 128, 17 * 27
            nn.ReLU(),
            nn.Conv2d(output_channels * 4, output_channels * 8, 3, stride= 2, padding= 0), # 256, 8 * 13
            nn.ReLU(),
        )
        self.large_scale = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 11, stride=2, padding=1),  # 32, 40 * 60
            nn.ReLU(),
            nn.Conv2d(output_channels, output_channels * 2, 9, stride=2, padding=0),  # 64, 16 * 26
            nn.ReLU(),
            nn.Conv2d(output_channels * 2, output_channels * 7, 7, stride=1, padding=0),  # 128, 10 * 20
            nn.ReLU(),
            nn.Conv2d(output_channels * 4, output_channels * 8, 3, stride=1, padding=0),  # 256, 8 * 18
            nn.ReLU(),
        )
        self.mul_scale_fc = nn.Linear(8 * 18 + 8 * 13, classes)
        '''
    def forward(self, x):
        output = self.single(x)
        output = output.view(output.size(0), -1) # flap
        output = self.single_fc(output)
        return output
    '''
    def forward_multi_scale(self, x):
        output1 = self.small_scale(x)
        output1 = output1.view(output1.size(0), -1)
        output2 = self.large_scale(x)
        output2 = output2.view(output2.size(0), -1)
        feature = torch.cat((output1, output2), 1).type(torch.FloatTensor)
        output = self.mul_scale_fc(feature, 90)
        return output
    '''
