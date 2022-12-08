#import library
import pandas as pd;
import numpy as np;

#load data
mall = pd.read_csv("https://raw.githubusercontent.com/shrk-sh-ioai/tubes-ppdp/main/mall-customer-dt/mall-customers.csv", sep=",", encoding='cp1252');

#checking & cleaning data
dfCheck = mall[mall.isna().any(axis=1)];
print("Data with NaN values :"); print(dfCheck);

print("Data with minimum values :"); print(mall.min());
print("Data with maximum values :"); print(mall.max());