import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pandas_datareader import data
from datetime import date
from sklearn import metrics
from matplotlib import pyplot


dfDb = pd.read_excel('veri.xlsx', sheet_name='Sheet1')
df = pd.read_excel('yagis_petrol.xlsx', sheet_name='antalya')


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
            '2021-01-01'
            ]

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
yShift = y.shift(periods=-1)
yShift = yShift.fillna(0)
y = round(yShift)

startPoint = 0
splint = 365  
#731 2021' e kadar
lastNum = len(x) - splint

xForDtrClass = pd.read_excel('xForDtrClassYeni.xlsx')
xForDtrReg = pd.read_excel('xForDtrRegYeni.xlsx')
xForKnn = pd.read_excel('xForKnnYeni.xlsx')
xForLogisticRegression = pd.read_excel('xForLogisticRegressionYeni.xlsx')
xForNn = pd.read_excel('xForNnYeni.xlsx')
xForRfc = pd.read_excel('xForRfcYeni.xlsx')
xForSvm = pd.read_excel('xForSvmYeni.xlsx')
xForXgb = pd.read_excel('xForXgbYeni.xlsx')


#KNN algoritması
def KnnPred (x,y,cut):
    xTrain = []
    xTest = []
    yTrain = []
    yTest = []
    xTrainSc = []
    xTestSc = []
    yPred = []
    for i in range (1,lastNum+1):

        xTrain.append(x.iloc[0:cut-1+i, :].values)    
        xTest.append(x.iloc[cut-1+i:cut+i, :].values)
        yTrain.append(y.iloc[0:cut-1+i, :].values)
        yTest.append(y.iloc[cut-1+i:cut+i, :].values)
                
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        xTrainSc.append(sc.fit_transform(xTrain[i-1]))
        xTestSc.append(sc.transform(xTest[i-1]))
                
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(n_neighbors=2, metric = 'minkowski') #n_neighbors komsu sayisi, metric, mesafe olcumu
        model.fit(xTrainSc[i-1], yTrain[i-1]) #fit edemiyor, y' dekis ayılar float olduğu için fit edemediği farkeidlid ve sayılar yuvarlandı
        yPred.append(model.predict(xTestSc[i-1])) 
    
    return yPred



yPredKnn = KnnPred(x,y,splint)
yPredKnnImportance = KnnPred(xForKnn,y,splint)

#DTRRegression
def DtrRegPred(x,y,cut):
    xTrain = []
    xTest = []
    yTrain = []
    yTest = []
    yPred = []
    
    for i in range (1,lastNum+1):
        xTrain.append(x.iloc[0:cut-1+i, :].values)    
        xTest.append(x.iloc[cut-1+i:cut+i, :].values)
        yTrain.append(y.iloc[0:cut-1+i, :].values)
        yTest.append(y.iloc[cut-1+i:cut+i, :].values)
    
        from sklearn.tree import DecisionTreeRegressor
        model = DecisionTreeRegressor(random_state=0)
        model.fit(xTrain[i-1], yTrain[i-1])
        yPred.append(model.predict(xTest[i-1])) 
        
    return yPred

yPredDtr = DtrRegPred(x,y,splint)
yPredImportanceDtr = DtrRegPred(xForDtrReg,y,splint)

#DTRClassifaciton
def DtrClassPred(x,y,cut):
    xTrain = []
    xTest = []
    yTrain = []
    yTest = []
    yPred = []
    for i in range (1,lastNum+1):
        xTrain.append(x.iloc[0:cut-1+i, :].values)    
        xTest.append(x.iloc[cut-1+i:cut+i, :].values)
        yTrain.append(y.iloc[0:cut-1+i, :].values)
        yTest.append(y.iloc[cut-1+i:cut+i, :].values)
    
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier(random_state=0)
        model.fit(xTrain[i-1], yTrain[i-1])
        yPred.append(model.predict(xTest[i-1])) 
        
    return yPred
 
yPredDtrClass = DtrClassPred(x,y, splint)
yPredImportanceDtrClass = DtrClassPred(xForDtrClass,y,splint)

#LOGISTIC REGRESSION
def LogisticRegressionPred(x,y,cut):
    xTrain = []
    xTest = []
    yTrain = []
    yTest = []
    yPred = []
    for i in range (1,lastNum+1):
        xTrain.append(x.iloc[0:cut-1+i, :].values)    
        xTest.append(x.iloc[cut-1+i:cut+i, :].values)
        yTrain.append(y.iloc[0:cut-1+i, :].values)
        yTest.append(y.iloc[cut-1+i:cut+i, :].values)
    
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr')
        model.fit(xTrain[i-1], yTrain[i-1])
        yPred.append(model.predict(xTest[i-1])) 
        
    return yPred
    
yPredLogisticRegression = LogisticRegressionPred(x,y, splint)
yPredImportanceLogisticRegression = LogisticRegressionPred(xForLogisticRegression,y, splint)

#SUPPORT VECTOR MACHINE
def SvmPred(x,y,cut):
    xTrain = []
    xTest = []
    yTrain = []
    yTest = []
    yPred = []
    for i in range (1,lastNum+1):
        xTrain.append(x.iloc[0:cut-1+i, :].values)    
        xTest.append(x.iloc[cut-1+i:cut+i, :].values)
        yTrain.append(y.iloc[0:cut-1+i, :].values)
        yTest.append(y.iloc[cut-1+i:cut+i, :].values)
    
        from sklearn import svm
        model = svm.LinearSVC()
        model.fit(xTrain[i-1], yTrain[i-1])
        yPred.append(model.predict(xTest[i-1])) 
        
    return yPred
    
yPredSvm = SvmPred(x,y, splint)
yPredImportanceSvm = SvmPred(xForSvm,y, splint)

#RANDOM FOREST CLASSIFIAR
def RfclassPred(x,y,cut):
    xTrain = []
    xTest = []
    yTrain = []
    yTest = []
    yPred = []
    for i in range (1,lastNum+1):
        xTrain.append(x.iloc[0:cut-1+i, :].values)    
        xTest.append(x.iloc[cut-1+i:cut+i, :].values)
        yTrain.append(y.iloc[0:cut-1+i, :].values)
        yTest.append(y.iloc[cut-1+i:cut+i, :].values)
    
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
        model.fit(xTrain[i-1], yTrain[i-1])
        yPred.append(model.predict(xTest[i-1])) 
        
    return yPred
    
yPredRfc = RfclassPred(x,y, splint)
yPredImportanceRfc = RfclassPred(xForRfc,y, splint)

#NEURAL NETWORK
def NnPred(x,y,cut):
    xTrain = []
    xTest = []
    yTrain = []
    yTest = []
    yPred = []
    for i in range (1,lastNum+1):
        xTrain.append(x.iloc[0:cut-1+i, :].values)    
        xTest.append(x.iloc[cut-1+i:cut+i, :].values)
        yTrain.append(y.iloc[0:cut-1+i, :].values)
        yTest.append(y.iloc[cut-1+i:cut+i, :].values)
    
        from sklearn.neural_network import MLPClassifier
        model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
        model.fit(xTrain[i-1], yTrain[i-1])
        yPred.append(model.predict(xTest[i-1])) 
        
    return yPred

yPredNn = NnPred(x,y, splint)
yPredImportanceNn = NnPred(xForNn,y, splint)

#XGB
def XgbPred(x,y,cut):
    xTrain = []
    xTest = []
    yTrain = []
    yTest = []
    yPred = []
    for i in range (1,lastNum+1):
        xTrain.append(x.iloc[0:cut-1+i, :].values)    
        xTest.append(x.iloc[cut-1+i:cut+i, :].values)
        yTrain.append(y.iloc[0:cut-1+i, :].values)
        yTest.append(y.iloc[cut-1+i:cut+i, :].values)
    
        import xgboost as xgb
        model = xgb.XGBClassifier(objective="binary:logistic")
        model.fit(xTrain[i-1], yTrain[i-1])
        yPred.append(model.predict(xTest[i-1])) 
        
    return yPred
    
def XgbHingePred(x,y,cut):
    xTrain = []
    xTest = []
    yTrain = []
    yTest = []
    yPred = []
    for i in range (1,lastNum+1):
        xTrain.append(x.iloc[0:cut-1+i, :].values)    
        xTest.append(x.iloc[cut-1+i:cut+i, :].values)
        yTrain.append(y.iloc[0:cut-1+i, :].values)
        yTest.append(y.iloc[cut-1+i:cut+i, :].values)
    
        import xgboost as xgb
        model = xgb.XGBClassifier(objective="binary:hinge")
        model.fit(xTrain[i-1], yTrain[i-1])
        yPred.append(model.predict(xTest[i-1])) 
        
    return yPred   
    

yPredXgb = XgbPred(x,y, splint)
yPredImportanceXgb = XgbPred(xForXgb,y, splint)
yPredXgbHinge = XgbHingePred(x,y, splint)
yPredImportanceXgbHinge = XgbHingePred(xForXgb,y, splint)

models = ['Knn - K-Nearest Neighbors ','Knn - K-Nearest Neighbors Importance',
          'Dtr - Decision Tree Regression', 'Dtr - Decision Tree Regression Importance',
          'Dtr-Classifier - Decision Tree Classifier','Dtr-Classifier - Decision Tree Classifier Importance',
          'LR - Logistic Regression','LR - Logistic Regression Importance',
          'Svm - Support Vector Machine','Svm - Support Vector Machine Importance',
          'RfClass - Random Forest Class','RfClass - Random Forest Class Importance',
          'Nn - Neural Network','Nn - Neural Network Importance',
          'XgbCls','XgbCls Importance',
          'XgbClsHinge','XgbClsHinge Importance']

predModels = [yPredKnn,yPredKnnImportance,
              yPredDtr,yPredImportanceDtr,
              yPredDtrClass,yPredImportanceDtrClass,
              yPredLogisticRegression,yPredImportanceLogisticRegression,
              yPredSvm,yPredImportanceSvm,
              yPredRfc,yPredImportanceRfc,
              yPredNn,yPredImportanceNn,
              yPredXgb,yPredImportanceXgb,
              yPredXgbHinge,yPredImportanceXgbHinge]
predModels = pd.DataFrame(predModels)
predModels = predModels.transpose()

r2Scores = []
accScores = []
mseScores = []


y_test = y.iloc[splint:, :].values
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

for i in range(len(models)):
    accScores.append('%s, Acc Score: %.5f' % (models[i],(accuracy_score(y_test, predModels[i].tolist()))))
    r2Scores.append('%s, R2 Score: %.5f' % (models[i],(r2_score(y_test, predModels[i].tolist()))))
    mseScores.append('%s, Mse Score: %.5f' % (models[i],(mean_squared_error(y_test, predModels[i].tolist()))))
    #confusionMatrix.append('%s, ConfusionMatrix: %.5f' % (models[i],(confusion_matrix(y_test, predModels[i]))))
    

from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

f1Scores = []
recallScores = []
precisionScores = []

averages = ['macro','micro','weighted', 'binary']
for i in range(len(models)):
    for j in range(len(averages)):
        f1Scores.append('%s, f1Score:%s %.5f' % (models[i],averages[j],(f1_score(y_test, predModels[i].tolist(), average=averages[j]))))
        recallScores.append('%s, recallScores:%s %.5f' % (models[i],averages[j],(recall_score(y_test, predModels[i].tolist(), average=averages[j]))))
        precisionScores.append('%s, precisionScores:%s %.5f' % (models[i],averages[j],(precision_score(y_test, predModels[i].tolist(), average=averages[j]))))



from sklearn.metrics import confusion_matrix
ConfusionKnn = confusion_matrix(y_test, yPredKnn)
ConfusionKnnImportance = confusion_matrix(y_test, yPredKnnImportance)
ConfusionDtr = confusion_matrix(y_test, yPredDtr)
ConfusionDtrImportance = confusion_matrix(y_test, yPredImportanceDtr)
ConfusionDtClass = confusion_matrix(y_test, yPredDtrClass)
ConfusionDtClassImportance = confusion_matrix(y_test, yPredImportanceDtrClass)
ConfusionLR = confusion_matrix(y_test, yPredLogisticRegression)
ConfusionLRImportance = confusion_matrix(y_test, yPredImportanceLogisticRegression)
ConfusionSvm = confusion_matrix(y_test, yPredSvm)
ConfusionSvmImportance = confusion_matrix(y_test, yPredImportanceSvm)
ConfusionRf = confusion_matrix(y_test, yPredRfc)
ConfusionRfImportance = confusion_matrix(y_test, yPredImportanceRfc)
ConfusionNn = confusion_matrix(y_test, yPredNn)
ConfusionNnImportance = confusion_matrix(y_test, yPredImportanceNn)
ConfusionXgbCls = confusion_matrix(y_test, yPredXgb)
ConfusionXgbClsImportance = confusion_matrix(y_test, yPredImportanceXgb)
ConfusionXgbClsHinge = confusion_matrix(y_test, yPredXgbHinge)
ConfusionXgbClsHingeImportance = confusion_matrix(y_test, yPredImportanceXgbHinge)







