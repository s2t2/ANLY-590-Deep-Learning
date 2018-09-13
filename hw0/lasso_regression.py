# -*- coding: utf-8 -*-
# @Author: Kornraphop Kawintiranon
# @Date:   2018-09-12 00:48:54
# @Last Modified by:   Kornraphop Kawintiranon
# @Last Modified time: 2018-09-13 13:02:10

__author__ = "Kornraphop Kawintiranon"
__email__ = "kornraphop.k@gmail.com"

import matplotlib.pyplot as plt

import sys, os
import numpy as np
import pandas as pd

from sklearn import linear_model

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

	# LASSO test with different alphas
	lasso = linear_model.Lasso(max_iter = 10000, normalize=True)

	# Try several alphas
	coefs = []
	alphas = np.logspace(-3, 3, 200)

	is_print = False
	for a in alphas:
		lasso.set_params(alpha=a)
		lasso.fit(X, Y)
		coefs.append(lasso.coef_)

		# Check the last three features left
		if len([ c for c in lasso.coef_ if c != 0 ]) == 3 and not is_print:
			last_three_feature = []
			for idx, c in enumerate([ c for c in lasso.coef_ if c != 0 ]):
				last_three_feature.append(list(X)[idx])
			print("The last three features are " + ", ".join(last_three_feature))	# AtBat, Hits, HmRun
			is_print = True

	# Plot alpha-coefficient relation
	plt.figure(figsize=(10, 6))
	ax = plt.gca()
	ax.plot(alphas, coefs)
	ax.set_xscale('log')
	plt.xlabel('alpha')
	plt.ylabel('weights')
	plt.axis('tight')
	plt.title('LASSO coefficients as a function of the regularization')

	# Find the optimal alpha by cross-validation
	lassocv = linear_model.LassoCV(alphas=None, cv=10, max_iter=100000, normalize=True)
	lassocv.fit(X, Y)
	opitmal_alpha = lassocv.alpha_

	# Build model using the optimal alpha
	lasso.set_params(alpha=opitmal_alpha)
	lasso.fit(X, Y)

	# Check coefficients
	print(lasso.coef_)
	print("Feature number left in the model: " + str(len([ x for x in lasso.coef_ if x != 0 ])))

	plt.show()

if __name__=="__main__":
	main()