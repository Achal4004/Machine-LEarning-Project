# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 11:56:43 2020

@author: hp
"""



    ######## A Healthcare Domain Chatbot to simulate the predictions of a General Physician ########
            ######## A pragmatic Approach for Diagnosis ############

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
train_dataset = pd.read_csv('Training.csv')
test_dataset = pd.read_csv('Testing.csv')

train_dataset.isnull().sum()

# Slicing and Dicing the dataset to separate features from predictions
X = train_dataset.iloc[:, 0:132].values
y = train_dataset.iloc[:, -1].values

# Dimensionality Reduction for removing redundancies
minimised_dataset = train_dataset.groupby(train_dataset['prognosis']).max()
minimised_dataset

# Encoding String values to integer constants
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)

#****************************************************************


# Splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Implementing the Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)


# Saving the information of columns
columns     = train_dataset.columns
columns     = columns[:-1]


# Checking the Important features
importances = classifier.feature_importances_
indices = np.argsort(importances)[::-1]
features = columns


#*******************************************************************


# Implementing the Visual Tree
from sklearn.tree import _tree

# Method to simulate the working of a Chatbot by extracting and formulating questions
def execute_healthbot():

    print("Please reply with yes/Yes or no/No for the following symptoms")
    def print_disease(node):
        #print(node)
        node = node[0]
        #print(len(node))
        value  = node.nonzero()
        #print(value)
        disease = labelencoder.inverse_transform(value[0])
        return disease

    def tree_to_code(tree, feature_names):
        tree_ = tree.tree_
        #print(tree_)
        feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]
        #print("def tree({}):".format(", ".join(feature_names)))
        symptoms_present = []
        
#**************************************************************************************************
        def recurse(node, depth):
            indent = "  " * depth
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]
                print(name + " ?")
                ans = input()
                ans = ans.lower()
                if ans == 'yes':
                    value = 1
                else:
                    value = 0
                if  value <= threshold:
                    recurse(tree_.children_left[node], depth + 1)
                else:
                    symptoms_present.append(name)
                    recurse(tree_.children_right[node], depth + 1)
                    
#**************************************************************************
                    
            else:
                present_disease = print_disease(tree_.value[node])
                print( "You may have " +  present_disease )
                print()
                red_columns = minimised_dataset.columns
                symptoms_given = red_columns[minimised_dataset.loc[present_disease].values[0].nonzero()]
                print("symptoms present  " + str(list(symptoms_present)))
                print()
                print("symptoms given "  +  str(list(symptoms_given)) )
                print()
                confidence_level = (1.0*len(symptoms_present))/len(symptoms_given)
                print("confidence level is " + str(confidence_level))
                print()
                
               # print('The model suggests:')
               # print()
                #row = doctors[doctors['disease'] == present_disease[0]]
               # print('Consult ', str(row['name'].values))
                #print()
                #print('Visit ', str(row['link'].values))
                        #print(present_disease[0])


        recurse(0, 1)

    tree_to_code(classifier,columns)





# Execute the bot and see it in Action
execute_healthbot()















































