#PLEASE WRITE THE GITHUB URL BELOW!
#

import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.svm import SVC


def load_dataset(dataset_path):
	return pd.read_csv(dataset_path)

def dataset_stat(dataset_df):	
	date_shape = dataset_df.shape
	target_size = dataset_df.groupby("target").size()
	num_class0 = target_size[0]
	num_class1 = target_size[1]
	return date_shape[1], num_class0, num_class1

def split_dataset(dataset_df, testset_size):
	X = dataset_df.drop(columns="target",axis=1)
	y = dataset_df["target"]
	X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=testset_size)
	return X_train, X_test, y_train, y_test

def decision_tree_train_test(x_train, x_test, y_train, y_test):
	dt_cls = DecisionTreeClassifier()
	dt_cls.fit(x_train,y_train)
	dt_acc = accuracy_score(y_test, dt_cls.predict(x_test))
	dt_prec = precision_score(y_test, dt_cls.predict(x_test))
	dt_recall = recall_score(y_test, dt_cls.predict(x_test))
	return dt_acc, dt_prec, dt_recall

def random_forest_train_test(x_train, x_test, y_train, y_test):
	rf_cls = RandomForestClassifier()
	rf_cls.fit(x_train, y_train)
	rf_acc = accuracy_score(y_test, rf_cls.predict(x_test))
	rf_prec = precision_score(y_test, rf_cls.predict(x_test))
	rf_rec = recall_score(y_test, rf_cls.predict(x_test))
	return rf_acc, rf_prec, rf_rec



def svm_train_test(x_train, x_test, y_train, y_test):
	svm_cls = SVC()
	svm_cls.fit(x_train, y_train)
	svm_acc = accuracy_score(y_test, svm_cls.predict(x_test))
	svm_prec = precision_score(y_test, svm_cls.predict(x_test))
	svm_recall = recall_score(y_test, svm_cls.predict(x_test))
	return svm_acc, svm_prec, svm_recall

def print_performances(acc, prec, recall):
	#Do not modify this function!
	print ("Accuracy: ", acc)
	print ("Precision: ", prec)
	print ("Recall: ", recall)

if __name__ == '__main__':
	#Do not modify the main script!
	data_path = sys.argv[1]
	data_df = load_dataset(data_path)

	n_feats, n_class0, n_class1 = dataset_stat(data_df)
	print ("Number of features: ", n_feats)
	print ("Number of class 0 data entries: ", n_class0)
	print ("Number of class 1 data entries: ", n_class1)

	print ("\nSplitting the dataset with the test size of ", float(sys.argv[2]))
	x_train, x_test, y_train, y_test = split_dataset(data_df, float(sys.argv[2]))

	acc, prec, recall = decision_tree_train_test(x_train, x_test, y_train, y_test)
	print ("\nDecision Tree Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = random_forest_train_test(x_train, x_test, y_train, y_test)
	print ("\nRandom Forest Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = svm_train_test(x_train, x_test, y_train, y_test)
	print ("\nSVM Performances")
	print_performances(acc, prec, recall)