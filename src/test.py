from statistics import mean

import numpy as np
import pandas as pd
from numpy import std
from numpy.linalg import eig
from sklearn.decomposition import PCA

path = r"C:\Users\wzc\Desktop\New 文本文档.txt"




def std_(x):
    x = x-x.mean()
    return x / np.std(x,ddof=1)


file_object = open(path)
finnal_list = []
line = file_object.readline()
while line:
    print(line)
    line = line.strip()
    line_list = line.split("\t")
    finnal_list.append(line_list)
    line = file_object.readline()
df = pd.DataFrame(finnal_list,dtype=float)

df = df.apply(std_,axis=0)

X_correlation = df.corr()
pca = PCA()
# model = pca.fit(X_correlation)
#
# model.components_*np.sqrt(model.explained_variance_.reshape(model.n_components_,1))
# X_correlation = X_correlation.values
eigenvalues,eigenvectors = eig(X_correlation)
print()
