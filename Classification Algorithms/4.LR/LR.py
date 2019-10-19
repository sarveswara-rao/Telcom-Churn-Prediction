# -*- coding: utf-8 -*-
"""
@author: sarveswara rao
"""
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt

def plots(x, y, label):
	plt.xlabel("No of features")
	plt.ylabel(label)
	plt.plot(x, y, marker = 'o')
	plt.grid(True)
	plt.show()

if __name__ == "__main__":

	df = pd.read_csv('D:\\bda_project\\Final\\Project1\\ibm_churn_prediction.csv')
	
	# dropping all the missng values rows
	missing = []
	for i in range(7043):
		try:
			np.float(df['TotalCharges'][i])
		except:
			missing.append(i)

	df = df.drop(missing)
	df = df.drop(columns = ['customerID'])
	
	# Changing the data-type of the total charges from object to float64
	df['TotalCharges'] = df['TotalCharges'].astype(np.float64)

	# Dividing the data into features set and target set
	X = df.iloc[:, 0:19]	
	y = df.iloc[:, 19]

	y = pd.DataFrame(data = y, columns={'Churn'})
    
	# Encoding of categorical values to numerical values
	y['Churn'] = LabelEncoder().fit_transform(y['Churn'])
	X['gender'] = LabelEncoder().fit_transform(X['gender'])
	X['Partner'] = LabelEncoder().fit_transform(X['Partner'])
	X['Dependents'] = LabelEncoder().fit_transform(X['Dependents'])
	X['PhoneService'] = LabelEncoder().fit_transform(X['PhoneService'])
	X['PaperlessBilling'] = LabelEncoder().fit_transform(X['PaperlessBilling'])

	# Using one hot encoding for features having more than two possible categories
	features = ['MultipleLines','InternetService','OnlineSecurity', 
            'OnlineBackup','DeviceProtection', 'TechSupport', 'StreamingTV',
            'StreamingMovies','Contract', 'PaymentMethod']
	
	X = pd.get_dummies(X, columns = features)
    
	#Importing the new data to a csv file :)
	X.to_csv('D:\\bda_project\\Final\\Project1\\new.csv')
	y.to_csv('D:\\bda_project\\Final\\Project1\\churn.csv')
	print(X.columns,'\n')
	# Standardizing the features
	col_names = ['tenure', 'MonthlyCharges', 'TotalCharges']
	features = X[col_names]
	features = StandardScaler().fit_transform(features)
	X[col_names] = features
	y = np.ravel(y)

	print("\n---------Features Vs Different evalution metrics---------\n")

	from sklearn.feature_selection import SelectKBest, f_classif

	k_values = []
	for i in range(X.shape[1]):
		k_values.append(i+1)

	p1 = []
	p2 = []
	p3 = []
	p4 = []
	p5 = []
	for i in range(1):
		# Everytime we are seleting best k features from the dataset
		test = SelectKBest(score_func = f_classif, k = i+1)
		X_test = test.fit_transform(X, y)
		X_new = pd.DataFrame(X_test)

		# using stratified k fold for equal class distribution in both training and test set
		accuracy = cross_val_score(LogisticRegression(), X_new, y, scoring = 'accuracy',
		 						cv = StratifiedKFold(5))
		precision = cross_val_score(LogisticRegression(), X_new, y,
						 scoring = 'precision', cv = StratifiedKFold(5))
		f1 = cross_val_score(LogisticRegression(), X_new, y, 
						scoring = 'f1', cv = StratifiedKFold(5))
		recall = cross_val_score(LogisticRegression(), X_new, y,
						scoring = 'recall', cv = StratifiedKFold(5))
		auc = cross_val_score(LogisticRegression(), X_new, y,
						scoring = 'roc_auc', cv = StratifiedKFold(5))

		p1.append(100*accuracy.mean())
		p2.append(100*precision.mean())
		p3.append(100*f1.mean())
		p4.append(100*recall.mean())
		p5.append(100*auc.mean())


	plots(k_values, p1, 'Accuracy')
	plots(k_values, p2, 'Precision')
	plots(k_values, p3, 'F1-Score')
	plots(k_values, p4, 'Recall')
	plots(k_values, p5, 'Area Under ROC')