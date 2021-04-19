import sys
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from numpy import loadtxt
from xgboost import XGBClassifier
import random
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from xgboost import plot_importance
from matplotlib import pyplot 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

def Train():
	#dataset = loadtxt('data_out_no_headers.csv', delimiter=",")
	dataset = loadtxt('3mo_no_headers.csv', delimiter=",")
	X = dataset[:,1:]
	Y = dataset[:,0]

	seed = random.randint(0,1000000)
	print(seed)

	rng = np.random.RandomState(seed)
	randSet = rng.random_integers(0,1000000, 1000)

	XGBAUCs = []
	LOGREGAUCs = []
	maxAUC = 0
	for iteration in range(0, 1000):
		print("\n\n\ncurrent iteration of 1000:", iteration, "\n\n\n")
		
		"""
		# Bootstrap

		indices = rng.random_integers(0, len(X) - 1, len(X))
		mask = np.ones(len(indices), dtype=bool)
		mask[indices,] = False
		X_train, X_test = X[indices], X[mask]
		y_train, y_test = Y[indices], Y[mask]

		"""

		# Random Train/Test Split

		rng = np.random.RandomState(randSet[iteration])
		X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=rng)
	
		eval_set = [(X_test, y_test)]
	
		"""
		# GridSearch for finding best XGB  hyperparameters

		opt_dict = {"max_depth": [1,2,3,5,7],
					"learning_rate": [0.01, 0.10, 0.25, 0.50],
					"n_estimators": [2000],
					"scale_pos_weight": [1,2,3,5],
					"reg_alpha": [0,1,3,5,7],
					"reg_lambda":[0,1,3,5,7],
					"early_stopping_rounds":[100], 
					"verbose":[True]
					}
	
		model = XGBClassifier(n_estimators=100)
	
		GridModel = GridSearchCV(model, opt_dict, scoring='roc_auc', verbose = 2, n_jobs=-1, cv=5)
	
		GridModel.fit(X_train, y_train)
	
		print(GridModel.best_params_)
	
		print("\n")
	
		y_predicted = model.predict(X_test)
		"""
	
		scaler = StandardScaler()
		X_trainP = pd.DataFrame(X_train)
		scaler.fit(X_trainP)
		X_scaled = pd.DataFrame(scaler.transform(X_trainP),columns = X_trainP.columns)
	
		logReg = LogisticRegression(random_state=seed)
		logReg.fit(X_train, y_train)
		logRegPred = logReg.predict(X_test)
		logRegProbs = logReg.predict_proba(X_test)[:,1]
	
		feature_importance = abs(logReg.coef_[0])
		feature_importance = 100.0 * (feature_importance / feature_importance.max())
		sorted_idx = np.argsort(feature_importance)[::-1]
	
		print("LogReg sorted features")
		print(sorted_idx)
		print("% Importance for first 5")
		for i in range(0, 5):
			print(feature_importance[sorted_idx[i]])
	
		model = XGBClassifier(scale_pos_weight=1, max_depth=2, n_estimators=10000, learning_rate=0.05 , reg_alpha=3, reg_lambda=1, random_state=seed)
		model.fit(X_train, y_train, eval_metric='auc', eval_set=eval_set, early_stopping_rounds=100, verbose=True)
	
		print("\n")
	
		y_predicted = model.predict(X_test)
		XGBProbs = model.predict_proba(X_test)[:,1]
		print(y_predicted)
		predictions = [round(value) for value in y_predicted]
		accuracy = accuracy_score(y_test, predictions)
		accLog = accuracy_score(y_test, logRegPred)
		print("XGB Accuracy: %.2f%%" % (accuracy * 100.0))
		print("LogReg Accuracy: %.2f%%" % (accLog * 100.0))
	
		print("\n")
	
		f = model._Booster.get_score(importance_type='weight')
		d = model._Booster.get_score(importance_type='gain')
	
		print("Gain Scores:")
		print(sorted(d.items(), key=lambda x: x[1], reverse=True))
		print("\n")
	
		print("test")
		print(max(d, key=d.get))
	
		print("Weight Scores:")
		print(sorted(f.items(), key=lambda x: x[1], reverse=True))
		print("\n")
	
		#Stats and Plots of predictions
	
		PrintConfusionMatrix("XGB", y_test, predictions)
		PrintConfusionMatrix("LogReg", y_test, logRegPred)
	
		print(model)
	
		data = pd.read_csv("data_out.csv", low_memory=False)
	
		colnames = list(data.columns)[1:]
	
		pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
		print("pyplot Feature Importance:")
		#pyplot.show()
	
		print("XGB plot_importance")
		plot_importance(model)
		#pyplot.show()
	
		print("XGB AUC:")
		XGBAUC = (roc_auc_score(y_test, XGBProbs))
		XGBAUCs.append(XGBAUC)
		print(roc_auc_score(y_test, XGBProbs))
	
		print("ROC CURVE, XGB")
		fpr, tpr, thresholds = roc_curve(y_test, XGBProbs)
		pyplot.plot([0, 1], [0, 1], linestyle='--')
		pyplot.plot(fpr, tpr, marker='.')
		pyplot.title("XGB ROC CURVE")
		#pyplot.show()
		pyplot.clf()

		#convert below to function later
	
		precision, recall, thresholds = precision_recall_curve(y_test, XGBProbs)
	
		f1 = f1_score(y_test, y_predicted)
		AUC = auc(recall, precision)
		ap = average_precision_score(y_test, XGBProbs)
		print('f1=%.3f auc=%.3f ap=%.3f' % (f1, AUC, ap))
	
		print("PR CURVE, XGB")
		pyplot.plot([0, 1], [0.1, 0.1], linestyle='--')
		pyplot.plot(recall, precision, marker='.')
		pyplot.title("XGB PR CURVE")
		#pyplot.show()
		pyplot.clf()
	
	
		print("LogReg AUC:")
		LOGREGAUCs.append(roc_auc_score(y_test, logRegProbs))
		print(roc_auc_score(y_test, logRegProbs))
	
		fpr, tpr, thresholds = roc_curve(y_test, logRegProbs)
		pyplot.plot([0, 1], [0, 1], linestyle='--')
		pyplot.plot(fpr, tpr, marker='.')
		pyplot.title("LOGREG ROC CURVE")
		#pyplot.show()
		pyplot.clf()
	
		precision, recall, thresholds = precision_recall_curve(y_test, logRegProbs)
	
		f1 = f1_score(y_test, logRegPred)
		AUC = auc(recall, precision)
		ap = average_precision_score(y_test, logRegProbs)
		print('f1=%.3f auc=%.3f ap=%.3f' % (f1, AUC, ap))
	
		print("PR CURVE, LOGREG")
		pyplot.plot([0, 1], [0.1, 0.1], linestyle='--')
		pyplot.plot(recall, precision, marker='.')
		pyplot.title("LOGREG PR CURVE")
		#pyplot.show()
		pyplot.clf()


		"""
		LRPred = logReg.predict_proba(X)[:,1]
		XGBPred = model.predict_proba(X)[:,1]
		print(LRPred)
		print(XGBPred)
		OutcomesToFile(XGBPred, LRPred, "90day1modelpredictions.csv")
		"""

		if XGBAUC > maxAUC:
			maxAUC = XGBAUC
			bestSeed = randSet[iteration]
			print("\n")
			print("New best:")
			print("AUC:", maxAUC)
			print("seed:", bestSeed)

	print(maxAUC, bestSeed)

	XGBSort = np.array(XGBAUCs)
	XGBSort.sort()
	cLower = XGBSort[int(0.025 * len(XGBSort))]
	cUpper = XGBSort[int(0.975 * len(XGBSort))]
	print("Confidence interval for  XGB: [{:0.3f} - {:0.3}]".format(cLower, cUpper))
	print("Average of", np.average(XGBAUCs))

	pyplot.hist(XGBAUCs, bins=50)
	pyplot.title('Bootstrapped XGB AUC scores')
	#pyplot.show()
	pyplot.clf()

	LRSort = np.array(LOGREGAUCs)
	LRSort.sort()
	cLower = LRSort[int(0.025 * len(LRSort))]
	cUpper = LRSort[int(0.975 * len(LRSort))]
	print("Confidence interval for  XGB: [{:0.3f} - {:0.3}]".format(cLower, cUpper))
	print("Average of", np.average(LOGREGAUCs))

	pyplot.hist(LOGREGAUCs, bins=50)
	pyplot.title('Bootstrapped LogReg AUC scores')
	#pyplot.show()
	pyplot.clf()



	"""

	# Generate Output for calibration plots

	TrainZeros = np.zeros((12090,1), dtype=int)
	TestOnes = np.ones((4030,1), dtype=int)
	
	y_train = y_train[np.newaxis]
	y_test = y_test[np.newaxis]

	TrainCSV = np.hstack((X_train, TrainZeros))
	TrainCSV = np.hstack((TrainCSV, y_train.T))
	TrainCSV = np.hstack((TrainCSV, TrainZeros))
	TrainCSV = np.hstack((TrainCSV, TrainZeros))
	
	TestCSV = np.hstack((X_test, TestOnes))
	TestCSV = np.hstack((TestCSV, y_test.T))

	logRegProbs = logRegProbs[np.newaxis]
	XGBProbs = XGBProbs[np.newaxis]

	TestCSV = np.hstack((TestCSV, logRegProbs.T))
	TestCSV = np.hstack((TestCSV, XGBProbs.T))

	FullOut = np.vstack((TestCSV, TrainCSV))

	np.savetxt("Test.csv", FullOut, delimiter=",")

	"""
	print("seed:", seed)

	return colnames

def ConvertToFeatName(colnames):
	A = input("Enter feature number: ")

	if A != "":
		print(A, colnames[int(A)])
		ConvertToFeatName(colnames)

def OutcomesToFile(X, L, fileName):
	outFile = open(fileName, 'w')
	for i in range(0, len(X)):
		outFile.write("%f,%f\n" % (X[i], L[i]))
	outFile.close()

def PrintConfusionMatrix(methodname, ytest, preds):
		TN, FP, FN, TP = confusion_matrix(ytest, preds).ravel()
		print(methodname, ":\nTN", TN, "\nFP", FP, "\nFN", FN, "\nTP", TP)
		print("\n")
	
		TPR = TP/(TP+FN)
		FNR = FN/(FN+TP)
	
		FPR = FP/(FP+TN)
		TNR = TN/(TN+FP)
	
		print("TPR", TPR*100)
		print("FNR", FNR*100)
		print("\n")
		print("TNR", TNR*100)
		print("FPR", FPR*100)
		print("\n")

if __name__ == "__main__":
	colnames = Train()
	#ConvertToFeatName(colnames)
