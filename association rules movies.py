# -*- coding: utf-8 -*-
"""
Created on Wed May 13 15:09:07 2022

@author: lalith kumar
"""

import pandas as pd
# import dataset
df = pd.read_csv('C:\python notes\ASSIGNMENTS\ASSOCIATION RULES\my_movies.csv')
df.shape
df.head()
df.describe()
df.info()

# removing object data.

movies=df.iloc[:,5:]
movies.shape
list(movies)

# importing Apriori Algorithm
from mlxtend.frequent_patterns import apriori,association_rules

# checking Most Frequent item sets based on support
frequent_itemsets = apriori(movies, min_support=0.05, use_colnames=True)
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

plt.scatter(rules['support'], rules['confidence'],c='orange')
plt.show()


import seaborn as sns
sns.scatterplot('support', 'confidence', data=rules, hue='antecedents')
plt.show()

# 3d visualization.
 
support=rules["support"]
confidence=rules["confidence"]
lift=rules["lift"]
fig1=plt.figure()
ax1=fig1.add_subplot(111, projection = '3d')
ax1.scatter(support,confidence,lift)
ax1.set_xlabel(["support"],c="green")
ax1.set_ylabel(["confidence"],c="orange")
ax1.set_zlabel(["lift"],c="blue")

#===================================================================


















