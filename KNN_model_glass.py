import os 
# changing the working directory.
os.chdir(r'C:\Users\madhu\OneDrive\Desktop\360 DigiTMG\DataScience\ASSIGNMENTS SOLVED BY ME\KNN Model')
import pandas as pd # it is used for Data Manipulation.
import numpy as np # used for numerical calculation.
from sklearn.compose import ColumnTransformer # used for different transformations in to specific columns
from sklearn.neighbors import KNeighborsClassifier # K nearest neighbors algorithm.
from sklearn.preprocessing import MinMaxScaler # it is use for preprocessing
from sklearn import metrics # it is used for model evaluvation.
import matplotlib.pyplot as plt # data visuvalization.
from sklearn.preprocessing import FunctionTransformer # within the pipeline we can use user defined function.
from sklearn.impute import SimpleImputer # to impute simple Imputer.
from sklearn.model_selection import train_test_split,GridSearchCV # It is used for spliting the dataset into test and train sets.
import sweetviz # Auto EDA
from imblearn.over_sampling import SMOTE # It is used to balance the imbalanced data set.
from imblearn.pipeline import Pipeline # it is used to make pipeline.
from sqlalchemy import create_engine # used for database connection.
from feature_engine.outliers import Winsorizer
import joblib
import pickle

engine = create_engine('mysql+pymysql://{}:{}@localhost/{}'.format('root','madhu123','glass_diagnosis_db'))# creating connection to database.

glass = pd.read_csv(r"C:\Users\madhu\OneDrive\Desktop\360 DigiTMG\DataScience\DATA SETS\glass.csv") # reading the test data Frame

#diagnosis = pd.read_csv(r"C:\Users\madhu\OneDrive\Desktop\360 DigiTMG\DataScience\CODES\KNN_Classifier\cancerdata.csv") # reading the test data Frame

#diagnosis.to_sql('cancer_tbl',con =engine,if_exists='replace',chunksize=1000,index = False)

glass.to_sql('glass_tbl',con = engine,if_exists = 'replace', chunksize = 1000,index = False)  # exporting the data to database

##### Getting Data from the Mysql database.
query = 'select * from glass_tbl;' # SQL query

glass = pd.read_sql_query(query,engine) # getting Data from database.

glass.info()
#glass['Type'] = glass.Type.astype('object')


#############Desritive Statistics/EDA#################

print(glass.describe() )# descriptive stastics.

report = sweetviz.analyze(glass) # Auto EDA.
report.show_html('glass.html') # render tp html page.
glass.isna().sum() # checking null values.

# Outliers checking.
# created the function to see outliers
def boxplot():
    glass.plot(kind = 'box' ,subplots = True,figsize = (15,7))
    plt.subplots_adjust(wspace = 0.75)
boxplot()
# Based on the boxplot we decided to do outliers treatment.
################Data Preprocessing#####################

num_cols = list(glass.select_dtypes(include = ['float64']).columns) # taking Numeric columns
num_cols
outliers_treatment = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Fe'] # columns for outliers treatment.

# Pipeline Creation for winsorization, standazation, scaling.
# created a function for removing outliers
# log transformation is not possible beacause the in some features 0 value is present.

def sqrt_trans(x):
    return np.power(x,1/5)

winsorizer = Pipeline([('winsorizer',Winsorizer(capping_method = 'iqr',tail='both',fold=1.5))])
winsorizer = winsorizer.fit(glass.loc[:,outliers_treatment])

joblib.dump(winsorizer,'winsorizer.pkl')

winsorizer = joblib.load('winsorizer.pkl')
glass[outliers_treatment] = winsorizer.transform(glass.loc[:,outliers_treatment])
glass.info()


preprocess_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')),
                     ('normalization',FunctionTransformer(func = sqrt_trans, validate = False)),
                     ('scaler', MinMaxScaler())
                     ])




preprocess_pipeline.fit(glass.loc[:,num_cols])

joblib.dump(preprocess_pipeline,'preprocess_pipeline.pkl')

preprocess_pipeline = joblib.load('preprocess_pipeline.pkl')
glass1 = pd.DataFrame(preprocess_pipeline.transform(glass.loc[:,num_cols]))
glass1.info()

glass['Type'].value_counts()

smote = SMOTE(random_state = 0)

# Transform the dataset
X,Y= smote.fit_resample(glass1, glass['Type'])

X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

###########Model Building###################
# The k value is taken based on K = squareroot of n/2 formula
knn = KNeighborsClassifier(n_neighbors=10)

KNN = knn.fit(X_train,y_train)

predict = KNN.predict(X_train)
metrics.accuracy_score(predict,y_train)
pd.crosstab(predict, y_train)

pred_test = KNN.predict(X_test)
metrics.accuracy_score(pred_test,y_test)

##############Model Building(Changing hyperparameter = K)###############
accu = []
for i in range(3,20,2):
    knn = KNeighborsClassifier(n_neighbors=i)
    KNN = knn.fit(X_train,y_train)
    accu_train = metrics.accuracy_score(KNN.predict(X_train),y_train)
    accu_test = metrics.accuracy_score(KNN.predict(X_test),y_test)
    accu.append([accu_train,accu_test])
    
plt.plot(range(3,20,2),[i[0] for i in accu],'ro-')
plt.plot(range(3,20,2),[i[1] for i in accu],'bo-')

# By Using the smote we observed that the accuracy score increases a lot. the right fit is K=7 because there no underfitting and overfitting.
# This Value i got after the SMOTE technique is use.

##############Model Building(Changing hyperparameter = K) BY GridSearchCV###############
k_range = list(np.arange(3,20,2))

param_grid = dict(n_neighbors = k_range)

grid = GridSearchCV(knn,param_grid,cv =5,scoring = 'accuracy',return_train_score = False, verbose = 1)

KNN_new = grid.fit(X_train,y_train)

print(KNN_new.best_params_)

accuracy = KNN_new.best_score_ *100
print("Accuracy for our training dataset with tuning is : {:.2f}%".format(accuracy) )


# Predict the class on test data
pred = KNN_new.predict(X_test)
pred

cm = metrics.confusion_matrix(y_test, pred)

metrics.accuracy_score(y_test,pred)

cmplot = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = glass['Type'])
cmplot.plot()
cmplot.ax_.set(title = 'Type of GLASS Detection - Confusion Matrix', 
               xlabel = 'Predicted Value', ylabel = 'Actual Value')

# Save the model
knn_best = KNN_new.best_estimator_
pickle.dump(knn_best, open('knn.pkl', 'wb'))

pickle.load()