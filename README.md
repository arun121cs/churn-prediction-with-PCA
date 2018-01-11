# churn-prediction-with-PCA
DImentionality reduction, machine learning algorithms, prediction


Steps:-
1.Import necessary libraries
2.Clean data
Data column CSA was dropped since it has a lot of Nan values and filling these values can give more noise to data.
4.Split data based on CALIBRAT values
3.Data Imputation
Missing values filled using mean.
4.Split CALIBRAT ==1 dataset further to apply models,split output variable,CHURNDEP from calibration data.
5.Apply PCA
PCA gives more than 99% of data information in just two dimentions.Two components selected.
6.Train models based on CALIBRAT ==1 data 
Since all models performed well for the given data
7.Apply same models on CALIBRAT ==0
8.Predict CHURNDEP for validation data and save output in dataframes.

 
Used models:-
1. Logistic regression
2. KNN
3.Decision Tree
4.Random Forest
5.Gaussian NB
6. SVM with rbf kernel


