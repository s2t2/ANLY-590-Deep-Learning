# -*- coding: utf-8 -*-
# @Author: Kornraphop Kawintiranon
# @Date:   2018-09-13 13:25:03
# @Last Modified by:   Kornraphop Kawintiranon
# @Last Modified time: 2018-09-13 15:47:13

__author__ = "Kornraphop Kawintiranon"
__email__ = "kornraphop.k@gmail.com"

import matplotlib.pyplot as plt

import sys, os
import numpy as np
import pandas as pd

from sklearn import linear_model
from sklearn.metrics import mean_squared_error

def main():
	# Read data from file 'Hitters.csv' 
	# (in the same directory that your python process is based)
	# Control delimiters, rows, column names with read_csv (see later)
	data = pd.read_csv("Hitters.csv") 

	# Preview the first 5 lines of the loaded data 
	print(data.head())

	# Extract only numeric columns
	numeric_col_idx = []
	for idx, t in enumerate(data.dtypes):
		if str(t) in ["int64", "float64"]:
			numeric_col_idx.append(idx)
	numeric_data = data.iloc[:,numeric_col_idx]

	# Remove NaN data
	numeric_data = numeric_data.dropna()

	# Preview the first 5 lines
	print(numeric_data.head())

	# Prepare data
	X = numeric_data.iloc[:,:-1]
	Y = numeric_data.iloc[:,-1]

	print("Record number: " + str(len(Y)))

	# Ridge test with different alphas
	ridge = linear_model.Ridge(normalize=True)

	# Try several alphas
	coefs = []
	alphas = np.logspace(-10, 10, 200)

	is_print = False
	for a in alphas:
		ridge.set_params(alpha=a)
		ridge.fit(X, Y)
		coefs.append(ridge.coef_)

		# Check the last three features left
		if len([ c for c in ridge.coef_ if c != 0 ]) == 3 and not is_print:
			last_three_feature = []
			for idx, c in enumerate([ c for c in ridge.coef_ if c != 0 ]):
				last_three_feature.append(list(X)[idx])
			print("The last three features are " + ", ".join(last_three_feature))	# AtBat, Hits, HmRun
			is_print = True
	if not is_print:
		print("There is no last three features, all features still are not zero")

	# Plot alpha-coefficient relation
	plt.figure(figsize=(10, 6))
	ax = plt.gca()
	ax.plot(alphas, coefs)
	ax.set_xscale('log')
	plt.xlabel('alpha')
	plt.ylabel('weights')
	plt.axis('tight')
	plt.title('Ridge coefficients as a function of the regularization')

	# Find the optimal alpha by cross-validation
	ridgecv = linear_model.RidgeCV(alphas=alphas, cv=10, normalize=True)
	ridgecv.fit(X, Y)
	opitmal_alpha = ridgecv.alpha_

	# Build model using the optimal alpha
	ridge.set_params(alpha=opitmal_alpha)
	ridge.fit(X, Y)

	# Check coefficients
	print(ridge.coef_)
	print("Feature number left in the model: " + str(len([ x for x in ridge.coef_ if x != 0 ])))

	# Check MSE
	y_pred = ridge.predict(X)
	mse = mean_squared_error(Y, y_pred)
	print("MSE: {0}".format(mse))	# 113323.37111528685

	plt.show()

if __name__=="__main__":
	main()