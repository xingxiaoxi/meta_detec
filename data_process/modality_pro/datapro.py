import pandas as pd
import numpy as np
data = pd.read_csv('../datasets/feature.csv')
word = data['Word']
data = data.drop(['Word'],axis=1)
word = list(word)
word = [s.lower() for s in word]
print(word,len(word))

numeric_features = data.dtypes[data.dtypes != 'object'].index
data[numeric_features] = data[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))
data[numeric_features] = data[numeric_features].fillna(0)
data = pd.get_dummies(data, dummy_na=True)
print(data.shape)

with open('../datasets/feature.txt','a+', encoding='utf-8') as f:
    for i in range(len(word)):
        f.write(word[i])
        for j in range(len(data.values[i])):
            f.write(' ')
            f.write(str(data.values[i][j]))
        f.write('\n')
