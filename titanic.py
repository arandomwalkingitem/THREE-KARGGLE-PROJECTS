
from sklearn.neural_network import MLPClassifier
import scipy.integrate as spi
import numpy as np
import matplotlib.pyplot as pl
import pandas as pd

#录入
#filename='total.csv'
#data=pd.read_csv(filename,encoding='utf-8')
#print(data)
filename1='test.csv'
filename='train.csv'
test=pd.read_csv(filename1,encoding='utf-8')#注意国内的csv文件，就算全是英文，encoding也要用utf-8,英文版的csv可以用utf-16
data=pd.read_csv(filename,encoding='utf-8')
#print(type(test))
#print(type(train))
#print(test)

#字符转数字
Gender_mapping = {'male':0, 'female':1}
data['Sex'] = data['Sex'].map(Gender_mapping)
Embarked_mapping = {'C':1, 'Q':2,'S':3}
data['Embarked'] = data['Embarked'].map(Embarked_mapping)
#print(data)

#缺失值处理
data['Age']=data['Age'].fillna(data['Age'].mean())
data['Fare']=data['Fare'].fillna(data['Fare'].mean())
data['Embarked']=data['Embarked'].fillna(data['Embarked'].mean())
#print(data)

#构建X y
# 检测数据中是否存在NaN,如果存在就返回True
y=data['Survived']
print(np.any(y.isnull())==True)
y=y.values.tolist()
X=data[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
print(np.any(X.isnull())==True)
X=X.values.tolist()

#分测试集和验证集
#random_state是个随机种子，确保每次随机分割得到相同的结果
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/4,random_state=7)
#print('训练集：')
#print(X_train)
#print(y_train)

#print('测试集：')
#print(X_test)
#print(y_test)

#用神经网络分类器进行训练
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                 hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X_train, y_train)
MLPClassifier(alpha=1e-05, hidden_layer_sizes=(5, 2), random_state=1,
              solver='lbfgs')

#预测After fitting (training), the model can predict labels for new samples:
output=clf.predict(X_test)
#setting an array element with a sequence.

clf.score(X_test,y_test)

#写入
filename='titanic_submission.csv'
df1=test['PassengerId']
df2 = pd.DataFrame(output, columns=['Survived'])
df=pd.concat([df1, df2],axis=1)
df.to_csv(filename,index=None,encoding='utf-8')#注意文件打开是没法写入的，注意要输出的DataFrame是df