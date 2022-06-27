# -*- coding: utf-8 -*-
"""
Created on Wed May  13 14:52:11 2022

@author: lalith kumar
"""

import pandas as pd

# import dataset
df = pd.read_csv('C:\\python notes\\ASSIGNMENTS\\ASSOCIATION RULES\\book.csv')
df.shape
df.describe()
list(df)
df.info()

# checking dummies
d=pd.get_dummies(df)
d.head()

# importing Apriori Algorithm
#pip install apyori
from mlxtend.frequent_patterns import apriori,association_rules

# checking Most Frequent item sets

frequent_itemsets = apriori(df, min_support=0.1, use_colnames=True)
frequent_itemsets

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.7)
rules
rules.sort_values('lift',ascending = False)

rules.sort_values('lift',ascending = False)[0:20]

rules[rules.lift>1]

# plot 
# histogram

rules[['support','confidence']].hist()

rules[['support','confidence','lift']].hist()

# scatter plot

import matplotlib.pyplot as plt

plt.scatter(rules['support'], rules['confidence'],c='red')
plt.show()


import seaborn as sns
sns.scatterplot('support', 'confidence', data=rules, hue='antecedents')

plt.show()
   

#------------------------------------------------------------




















