import sys
import datetime
import pandas as pd
import cx_Oracle
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pandas_datareader import data
import re
import time
from datetime import date
#import holidays
from sklearn import metrics
from matplotlib import pyplot
from sklearn.datasets import make_regression #feature importance icin gerekli
from sklearn.datasets import make_classification #feature importance icin gerekli


dfDb = pd.read_excel('veri.xlsx', sheet_name='Sheet1')


df = pd.read_excel('yagis_petrol.xlsx', sheet_name='antalya')

# df_new = pd.merge(dfDb, df, on='tarih', how='outer')



dates = df.iloc[:, 0:1]
datesValues = dates.values.tolist()
countDayWeek = 0
daysOfWeek = []
while countDayWeek < len(dates):
    daysOfWeek.append((pd.Timestamp(datesValues[countDayWeek][
                                        0]).dayofweek))  # GUN ADI eklemek icin , pd.Timestamp(abc[count][0]).day_name son guncellemede day_name calismamaktadir.
    countDayWeek = countDayWeek + 1  # 0 = pazartesi

countDayYear = 0
daysOfYear = []
while countDayYear < len(dates):
    daysOfYear.append((pd.Timestamp(datesValues[countDayYear][0]).dayofyear))  # Yilinkacinci gunu
    countDayYear = countDayYear + 1

countDayMonth = 0
daysOfMonth = []
while countDayMonth < len(dates):
    daysOfMonth.append((pd.Timestamp(datesValues[countDayMonth][0]).day))  # Ayin kacinin gunu
    countDayMonth = countDayMonth + 1

# ayın ilk günü
firstdayOfMonth = []
for firstday in daysOfMonth:
    if firstday == 1:
        firstdayOfMonth.append(1)
    else:
        firstdayOfMonth.append(0)

weekendOfWeek = []
for weekend in daysOfWeek:
    if weekend == 5 or weekend == 6:
        weekendOfWeek.append(1)
    else:
        weekendOfWeek.append(0)

# tatiller
holidays = ['2019-01-01', '2019-04-23', '2019-05-01', '2019-05-19',
            '2019-06-03', '2019-06-04', '2019-06-05', '2019-06-06', 
            '2019-07-15', '2019-08-10', '2019-08-11', '2019-08-12',
            '2019-08-13', '2019-08-14', '2019-08-30', '2019-10-28',
            '2019-10-29',
            
            
            '2020-01-01', '2020-04-23', '2020-05-01', '2020-05-19',
            '2020-05-23', '2020-05-24', '2020-05-25', '2020-05-26',
            '2020-07-15', '2020-07-30', '2020-07-31', '2020-08-01',
            '2020-08-02', '2020-08-03', '2020-08-30', '2020-10-28',
            '2020-10-29',
            '2021-01-01']

holidaysDatetime = []
for i in range(len(holidays)):
    holidaysDatetime.append(datetime.datetime.strptime(holidays[i], '%Y-%m-%d'))  # convert string to date
    
dfHolidays = pd.DataFrame(holidaysDatetime, columns=['Value'])
dfHolidays.columns = ['tarih']

dfDaysOfMonth = pd.DataFrame(daysOfMonth, columns=['gunler_ay'])
dfDaysOfWeek = pd.DataFrame(daysOfWeek, columns=['gunler_hafta'])
dfDaysOfYear = pd.DataFrame(daysOfYear, columns=['gunler_yil'])
dfFirstOfMonth = pd.DataFrame(firstdayOfMonth, columns=['ilkgun_ay'])
dfWeekendOfWeek = pd.DataFrame(weekendOfWeek, columns=['haftasonu'])
allData = pd.concat([df, dfDaysOfMonth, dfDaysOfWeek, dfDaysOfYear, dfFirstOfMonth, dfWeekendOfWeek], axis=1)

allData["holiday"] = allData["tarih"].isin(dfHolidays["tarih"])

hol = []
for holDay in allData["holiday"]:
    if holDay == True:
        hol.append(1)
    else:
        hol.append(0)

allData["holiday"] = hol

pivot_table = dfDb.pivot_table(index=["tarih"], columns=["uye_no"], values="miktar").fillna(0)
dfData = allData.set_index("tarih").join(pivot_table, lsuffix='_caller', rsuffix='_other', on='tarih').fillna(0)


# print (allData['tarih'].dtype) tip kontrol

# keras ile cnn modeli kurup bir sonraki günün tahminini 

litre = dfData.iloc[:, 1:2].values
y = dfData.iloc[:, 0:1].values
x = dfData.iloc[:, 2:].values
x = pd.DataFrame(x)
litre = pd.DataFrame(litre)
litreShift = litre.shift(periods = 1)
litreShift = litreShift.fillna(0)
y = pd.DataFrame(y)
yNormal = y
yShift = y.shift(periods=-1)
yShift = yShift.fillna(0)
y = round(yShift)

'''
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.33, random_state=0)   
'''

#xtrain 364 olacak
#xtest 365 olacak
#y train 364 olacak
#ytest 365 olacak

startPoint = 0
splint = 731    
#731 2021' e kadar
x_train = x.iloc[0:splint, :].values
x_test = x.iloc[splint:, :].values
y_train = y.iloc[0:splint, :].values
y_test = y.iloc[splint:, :].values

dfXTest = pd.DataFrame(x_test)
litreXTest = dfXTest.iloc[:,4:5]


xForDtrClass = pd.read_excel('xForDtrClassYeni.xlsx')
xForDtrReg = pd.read_excel('xForDtrRegYeni.xlsx')
xForKnn = pd.read_excel('xForKnnYeni.xlsx')
xForLogisticRegression = pd.read_excel('xForLogisticRegressionYeni.xlsx')
xForNn = pd.read_excel('xForNnYeni.xlsx')
xForRfc = pd.read_excel('xForRfcYeni.xlsx')
xForSvm = pd.read_excel('xForSvmYeni.xlsx')
xForXgb = pd.read_excel('xForXgbYeni.xlsx')





from sklearn.model_selection import GridSearchCV
'''
, 'min_samples_split': np.arange(1, 50),
                  'min_samples_leaf': np.arange(1, 50)
'''
'''
def dtreeClassGridSearch(X,y,nfolds):
    #create a dictionary of all values we want to test
    parameters = { 'criterion':['gini','entropy'],'max_depth': np.arange(1, 6),
                  'splitter' :['best', 'random'],'min_samples_split': np.arange(1, 6),
                  'min_samples_leaf':np.arange(1, 6),'max_features' : ['auto', 'sqrt', 'log2'],
                  'max_leaf_nodes' :np.arange(1, 6)
                  
                  }
    # decision tree model
    from sklearn.tree import DecisionTreeClassifier
    dtree_model=DecisionTreeClassifier()
    #use gridsearch to test all values
    model = GridSearchCV(dtree_model, parameters, cv=nfolds)
    #fit model to data
    model.fit(X, y)
    return model.best_params_

HyperParamsDtrClass = dtreeClassGridSearch(x,y,5)
HyperParamsDtrClassImportance = dtreeClassGridSearch(xForDtrClass,y,5)

def dtreeRegGridSearch(X,y,nfolds):
    #create a dictionary of all values we want to test
    parameters = { 'criterion':['mse', 'friedman_mse', 'mae', 'poisson'],'random_state': np.arange(1, 20),
                  'splitter' :['best', 'random']
                  }
    # decision tree model
    
    from sklearn.tree import DecisionTreeRegressor
    dtree_model=DecisionTreeRegressor()
    #use gridsearch to test all values
    model = GridSearchCV(dtree_model, parameters, cv=nfolds)
    #fit model to data
    model.fit(X, y)
    return model.best_params_

HyperParamsDtrReg = dtreeRegGridSearch(x,y,5)
HyperParamsDtrRegImportance = dtreeRegGridSearch(xForDtrReg,y,5)

def knnGridSearch (X,y):
    from sklearn.neighbors import KNeighborsClassifier
    parameters = {'n_neighbors':[4,5,6,7],
                  'leaf_size':[1,3,5],
                  'algorithm':['auto', 'kd_tree'],
                  'n_jobs':[-1]}
    

    algorithm=KNeighborsClassifier()
    #use gridsearch to test all values
    model = GridSearchCV(algorithm, parameters)
    #fit model to data
    model.fit(X, y)
    return model.best_params_

HyperParamsKnn = knnGridSearch(x,y)
HyperParamsKnnImportance = knnGridSearch(xForKnn,y)

def LrGridSearch (X,y):
    from sklearn.linear_model import LogisticRegression
    parameters = {"C":np.geomspace(1e-5,1e5, num=20), "penalty":["l1","l2"],
                  'solver': ['newton-cg', 'lbfgs', 'liblinear']}# l1 lasso l2 ridge
    
    algorithm=LogisticRegression()
    #use gridsearch to test all values
    model = GridSearchCV(algorithm, parameters,cv=10)
    #fit model to data
    model.fit(X, y)
    return model.best_params_

HyperParamsLr = LrGridSearch(x,y)
HyperParamsLrImportance = LrGridSearch(xForLogisticRegression,y)

def NnGridSearch (X,y,nfolds):
    from sklearn.neural_network import MLPClassifier
    parameters = {
    'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
    }
    
    algorithm=MLPClassifier()
    #use gridsearch to test all values
    model = GridSearchCV(algorithm, parameters, n_jobs=-1, cv=nfolds)
    #fit model to data
    model.fit(X, y)
    return model.best_params_

HyperParamsNn = NnGridSearch(x,y,2)
HyperParamsNnImportance = NnGridSearch(xForNn,y,2)

def RfclassGridSearch (X,y,nfolds):
    from sklearn.ensemble import RandomForestClassifier
    parameters = {
    'n_estimators': [50, 150, 250],
    'max_features': ['sqrt', 0.25, 0.5, 0.75, 1.0],
    'min_samples_split': [2, 4, 6]
    }
    
    algorithm=RandomForestClassifier()
    #use gridsearch to test all values
    model = GridSearchCV(algorithm, parameters, cv=nfolds)
    #fit model to data
    model.fit(X, y)
    return model.best_params_

HyperParamsRfclass = RfclassGridSearch(x,y,5)
HyperParamsRfclassImportance = RfclassGridSearch(xForRfc,y,5)


def SvmGridSearch (X,y):
    from sklearn.svm import LinearSVC 
    parameters = {'C': [0.1,1, 10, 100], "penalty":["l1","l2"], "multi_class":["ovr", "crammer_singer"]}
    
    #algorithm=svm.LinearSVC()
    #use gridsearch to test all values
    model = GridSearchCV(LinearSVC(),parameters,refit=True,verbose=2)
    #fit model to data
    model.fit(X, y)
    return model.best_params_

HyperParamsSvm = SvmGridSearch(x,y)
HyperParamsSvmImportance = SvmGridSearch(xForSvm,y)
'''
def XgbGridSearch (X,y):
    import xgboost as xgb
    parameters = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5],
        'objective': ['binary:logistic', 'binary:hinge']
        }
    
    algorithm=xgb.XGBClassifier()
    model = GridSearchCV(algorithm, parameters)
    model.fit(X, y)
    return model.best_params_

HyperParamsXgb = XgbGridSearch(x,y)
HyperParamsXfbImportance = XgbGridSearch(xForXgb,y)
'''
def XgbGridSearch (X,y):
    import xgboost as xgb
    parameters = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }
    
    algorithm=xgb.XGBClassifier(objective="binary:hinge")
    #use gridsearch to test all values
    model = GridSearchCV(algorithm, parameters)
    #fit model to data
    model.fit(X, y)
    return model.best_params_

HyperParamsXgbHinge = XgbGridSearch(x,y)
HyperParamsXfbHingeImportance = XgbGridSearch(xForXgb,y)

'''