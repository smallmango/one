import pandas as pd
import numpy as np
import pylab as pl



xlsx_file = pd.ExcelFile("./Absolute_error.xlsx")
data = xlsx_file.parse('Sheet1')

list_GPR_10 = data['GPR(K=10)'].values
list_GPR_100 = data['GPR(K=100)'].values
list_GPR_1000 = data['GPR(K=1000)'].values
list_SVR_LN = data['SVR(Lin)'].values
list_SVR_GU = data['SVR(Gau)'].values
list_MLG = data['MLG'].values
list_OPLDA = data['OPLDA'].values
list_OPMFA = data['OPMFA'].values

x_len = len(list_GPR_10)
np_x = np.arange(x_len)

pl.plot(np_x, list_MLG, 'r')

pl.xlabel('Absolute error[year]')
pl.ylabel('Cumulative score[%]')

pl.xlim(0.0, 20.0)
pl.ylim(0.0, 100.0)
pl.show()

