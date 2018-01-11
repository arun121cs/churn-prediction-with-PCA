
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd  
from sklearn.decomposition import PCA


# In[2]:


df = pd.read_csv('churn_data.csv')


# In[3]:


df.drop('CSA',axis=1,inplace = True) 


# In[14]:


train = df[df['CALIBRAT']==1]
test = df[df['CALIBRAT']==0]
train.is_copy = False
test.is_copy = False


# In[15]:


y_train = train['CHURNDEP']
y_train.is_copy = False


# In[9]:


train.drop('CHURNDEP',axis =1,inplace = True)


# In[10]:


X_train = train


# In[11]:


X_train.is_copy


# In[13]:


y_test = test['CHURNDEP']
y_test.is_copy = False


# In[16]:


test.drop('CHURNDEP',axis =1,inplace = True)


# In[17]:


X_test = test


# In[19]:


X_test.is_copy


# In[20]:


X_train.isnull().sum().sum()


# In[21]:


X_train.fillna(X_train.mean(),inplace=True)


# In[22]:


X_test.fillna(X_test.mean(),inplace=True)


# In[24]:


y_train.isnull().sum().sum() # NO NEED TO FILL SINCE NO NULL VALUES


# In[25]:


#DATA IS READY AFTER IMPUTATION AND IT HAS BEEN DIVIDED INTO TWO PARTS ACCORDING TO CALIBRAT FEATURE


# In[26]:


X_train.shape


# In[27]:


#SINCE THERE ARE 77 FEATURES,DATA NEEDS TO BE CONSIDERED IN RELATIVELY LESS DIMENTIONS 
#HERE I HAVE USED PCA FOR DIMENTIONALITY REDUCTION.
#First i have splitted the training data further into  train_cal and test_cal


# In[28]:


from sklearn.model_selection import train_test_split 


# In[29]:


X_train_cal, X_test_cal, y_train_cal, y_test_cal = train_test_split(X_train,y_train , test_size=0.25, random_state=0)


# In[30]:


#PCA


# In[31]:


pca = PCA(n_components = 30,whiten = True)


# In[32]:


X_train_calP = pca.fit(X_train_cal).transform(X_train_cal)


# In[33]:


X_test_calP = pca.fit(X_test_cal).transform(X_test_cal)


# In[35]:


pca.explained_variance_ratio_


# In[36]:


#Since most of the variance is in first two dimentions, n=2 will be selected


# In[37]:


pca = PCA(n_components = 2,whiten = True)


# In[38]:


X_train_calP = pca.fit(X_train_cal).transform(X_train_cal)
X_test_calP = pca.fit(X_test_cal).transform(X_test_cal)


# In[39]:


pca.explained_variance_ratio_


# In[40]:


# Now the data is preprocessed and since PCA has inbuilt standardization,no need to normalize the data further.


# In[41]:


#Logistic regression model on calibration data


# In[42]:


from sklearn.linear_model import LogisticRegression


# In[43]:


lg = LogisticRegression(C=1e5)
lg.fit(X_train_calP,y_train_cal)


# In[46]:


y_pred_lg = lg.predict(X_test_calP)


# In[47]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test_cal, y_pred_lg)


# In[48]:


print('confusion_matrix')
print(cm)


# In[49]:


from sklearn.metrics import accuracy_score as ac


# In[50]:


accuracy_logistic_calibration =ac(y_test_cal, y_pred_lg)


# In[54]:


print('Accuracy')
print(accuracy_logistic_calibration)


# In[102]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn.fit(X_train_calP,y_train_cal)

y_pred_knn = knn.predict(X_test_calP)
cm_knn = confusion_matrix(y_test_cal, y_pred_knn)
accuracy_knn = ac(y_test_cal,y_pred_knn)
print('confusion_matrix')
print(cm_knn)
print('accuracy')
print(accuracy_knn)


# In[56]:


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train_calP,y_train_cal)
y_pred_nb = nb.predict(X_test_calP)
cm_nb = confusion_matrix(y_test_cal, y_pred_nb)
accuracy_nb = ac(y_test_cal,y_pred_nb)
print('confusion_matrix')
print(cm_nb)
print('accuracy')
print(accuracy_nb)


# In[58]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
dt.fit(X_train_calP,y_train_cal)
y_pred_dt = dt.predict(X_test_calP)
cm_dt = confusion_matrix(y_test_cal, y_pred_dt)
accuracy_dt = ac(y_test_cal,y_pred_dt)
print('confusion_matrix')
print(cm_dt)
print('accuracy')
print(accuracy_dt)


# In[60]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
rf.fit(X_train_calP,y_train_cal)
y_pred_rf = rf.predict(X_test_calP)
cm_rf = confusion_matrix(y_test_cal, y_pred_rf)
accuracy_rf = ac(y_test_cal,y_pred_rf)
print('confusion_matrix')
print(cm_rf)
print('accuracy')
print(accuracy_rf)


# In[63]:


from sklearn.svm import SVC
svc = SVC(kernel = 'rbf', random_state = 0)
svc.fit(X_train_calP, y_train_cal)
y_pred_svm = svc.predict(X_test_calP)
cm_svm = confusion_matrix(y_test_cal,y_pred_svm)
accuracy_svm = ac(y_test_cal,y_pred_svm)
print('confusion_matrix')
print(cm_svm)
print('accuracy')
print(accuracy_svm)


# In[64]:


print('Fitting for Validation data')


# In[65]:


x_train = pca.fit(X_train).transform(X_train)
x_test = pca.fit(X_test).transform(X_test)


# In[66]:


pca.explained_variance_ratio_


# In[101]:


#Logistic_Regression
lg_validation = LogisticRegression(C=1e5)
lg_validation.fit(x_train,y_train)
y_pred_lg_validation = lg_validation.predict(x_test)

#just to cheak how much variation is there with previous fit ##
y_pred_lg = lg.predict(x_test)
cm_lg_out = confusion_matrix(y_pred_lg_validation,y_pred_lg)
cm_lg_out


# In[95]:


y_pred_lg_validation.shape 


# In[71]:


y_pred_lg.mean()


# In[72]:


y_train.mean()


# In[96]:


output = pd.DataFrame({'churndep':y_pred_lg_validation.tolist()})
output.to_csv('output_logistic.csv')


# In[103]:


#KNN
from sklearn.neighbors import KNeighborsClassifier
knn_validation = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn_validation.fit(x_train,y_train)
y_pred_knn_validation = knn_validation.predict(x_test)


# In[104]:


output = pd.DataFrame({'churndep':y_pred_knn_validation.tolist()})
output.to_csv('output_Knn.csv')


# In[99]:


#just to cheak how much variation is there with previous fit ##
y_pred_knn = knn.predict(x_test)
cm_knn_out = confusion_matrix(y_pred_knn_validation,y_pred_knn)
cm_knn_out


# In[77]:


#GaussianNB
nb_validation = GaussianNB()
nb_validation.fit(x_train,y_train)
y_pred_nb_validation = nb_validation.predict(x_test)


# In[100]:


output = pd.DataFrame({'churndep':y_pred_nb_validation.tolist()})
output.to_csv('output_GaussianNB_classifier.csv')


# In[81]:


#just to cheak how much variation is there with previous fit ##
y_pred_nb = nb.predict(x_test)
accuracy_out = ac(y_pred_nb_validation,y_pred_nb)


# In[83]:


#Decision_Tree
dt_validation = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
dt_validation.fit(x_train,y_train)
y_pred_dt_validation = dt_validation.predict(x_test)


# In[84]:


output = pd.DataFrame({'churndep':y_pred_dt_validation.tolist()})
output.to_csv('output_decision_tree.csv')


# In[87]:


#just to cheak how much variation is there with previous fit ##
y_pred_dt = dt.predict(x_test)
cm_dt = confusion_matrix(y_pred_dt_validation, y_pred_dt)
accuracy_dt = ac(y_pred_dt_validation,y_pred_dt)
accuracy_dt


# In[88]:


#Random_forest
rf_validation = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
rf_validation.fit(x_train,y_train)
y_pred_rf_validation = rf_validation.predict(x_test)


# In[89]:


output = pd.DataFrame({'churndep':y_pred_rf_validation.tolist()})
output.to_csv('output_random_forest.csv')


# In[90]:


y_pred_rf = rf.predict(x_test)
cm_rf = confusion_matrix(y_pred_rf_validation, y_pred_rf)
accuracy_rf = ac(y_pred_rf_validation,y_pred_rf)
accuracy_rf


# In[91]:


#Support_Vector_Machine
from sklearn.svm import SVC
svc_validation = SVC(kernel = 'rbf', random_state = 0)
svc_validation.fit(x_train,y_train)
y_pred_svc_validation = svc_validation.predict(x_test)


# In[92]:


output = pd.DataFrame({'predicted_by_SVM_churndep':y_pred_svc_validation.tolist()})
output.to_csv('output_SVM.csv')


# In[93]:


y_pred_svc = svc.predict(x_test)
cm_svc = confusion_matrix(y_pred_svc_validation, y_pred_svc)
accuracy_svc = ac(y_pred_svc_validation,y_pred_svc)
accuracy_svc

