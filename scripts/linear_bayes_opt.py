import argparse

# Instantiate the parser
parser = argparse.ArgumentParser(description='Analyzing Bayesian Optimization Results')

# Required positional argument
parser.add_argument("-f", '--file', type=str,
                    help='Json file containing the results of the bayesian optimization')

args = parser.parse_args()


import json

Y = []
X = []
with open(args.file) as f:
    for line in f:
        data = json.loads(line)
        Y.append(data['target'])
        X.append(list(data['params'].values()))

print ("x_keys:", data['params'].keys())
# returns JSON object as
# a dictionary
#print (X)


import numpy as np
from sklearn.linear_model import LinearRegression
X = np.array(X)
# y = 1 * x_0 + 2 * x_1 + 3
y = np.array(Y)
reg = LinearRegression().fit(X, y)
reg.score(X, y)

print (reg.coef_ )
