import numpy as np # 线性代数包
import pandas as pd # 数据处理包
import matplotlib.pyplot as plt # 数据可视化包
import seaborn as sns # 数据可视化包
import sklearn
print(sklearn.__version__)
!pip3 install -U sklearn #升级

# 训练集
train_df = pd.read_csv("C:/Users/sl6118919/Python_Learn/超思机器学习/kaggle/hourse/train.csv")
print(train_df.shape)
train_df.head()

# 测试集
test_df = pd.read_csv("C:/Users/sl6118919/Python_Learn/超思机器学习/kaggle/hourse/test.csv")
print(test_df.shape)
test_df.head()

# 提交文件的示例
submit_example = pd.read_csv("C:/Users/sl6118919/Python_Learn/超思机器学习/kaggle/hourse/sample_submission.csv")
print(submit_example.shape)
submit_example.head()

train_df.describe()

3.数据预处理

#1.字符转数字 2.特征归一化 3.特征工程找向量 4.剔除异常值（两端5%极值）



# 3.2 类别型特征转换为数值型特征
train_df['2'].replace(['NA','C (all)','FV','RH','RL','RM'],[0,1,2,3,4,5],inplace=True)
test_df['2'].replace(['NA','C (all)','FV','RH','RL','RM'],[0,1,2,3,4,5],inplace=True)
train_df['5'].replace(['Grvl','Pave'],[0,1],inplace=True)
test_df['5'].replace(['Grvl','Pave'],[0,1],inplace=True)
train_df['6'].replace(['NA','Grvl','Pave'],[0,1,2],inplace=True)
test_df['6'].replace(['NA','Grvl','Pave'],[0,1,2],inplace=True)

train_df['7'].replace(['IR1','IR2','IR3','Reg'],[0,1,2,3],inplace=True)
test_df['7'].replace(['IR1','IR2','IR3','Reg'],[0,1,2,3],inplace=True)

train_df['8'].replace(['Bnk','HLS','Low','Lvl'],[0,1,2,3],inplace=True)
test_df['8'].replace(['Bnk','HLS','Low','Lvl'],[0,1,2,3],inplace=True)

train_df['9'].replace(['AllPub','NoSeWa'],[0,1],inplace=True)
test_df['9'].replace(['AllPub','NoSeWa'],[0,1],inplace=True)

train_df['10'].replace(['Corner','CulDSac','FR2','FR3','Inside'],[0,1,2,3,4],inplace=True)
test_df['10'].replace(['Corner','CulDSac','FR2','FR3','Inside'],[0,1,2,3,4],inplace=True)

train_df['11'].replace(['Gtl','Mod','Sev'],[0,1,2],inplace=True)
test_df['11'].replace(['Gtl','Mod','Sev'],[0,1,2],inplace=True)

train_df['12'].replace([
    'OldTown','BrkSide','SWISU','Edwards','Mitchel','IDOTRR','SawyerW',
    'Crawfor','Gilbert','ClearCr','Veenker','Sawyer','CollgCr','Blueste','NAmes',
    'StoneBr','NridgHt','Timber','NoRidge','MeadowV','BrDale','NPkVill','Blmngtn','Somerst','NWAmes'],
    [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24],inplace=True)
test_df['12'].replace([
    'OldTown','BrkSide','SWISU','Edwards','Mitchel','IDOTRR','SawyerW',
    'Crawfor','Gilbert','ClearCr','Veenker','Sawyer','CollgCr','Blueste','NAmes',
    'StoneBr','NridgHt','Timber','NoRidge','MeadowV','BrDale','NPkVill','Blmngtn','Somerst','NWAmes'],
    [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24],inplace=True)

train_df['13'].replace(['Norm','Artery','Feedr','PosA','RRAn','PosN',
'RRNn',
'RRNe',
'RRAe'],[0,1,2,3,4,5,6,7,8],inplace=True)
test_df['13'].replace(['Norm',
'Artery',
'Feedr','PosA','RRAn',
'PosN','RRNn','RRNe',
'RRAe'],[0,1,2,3,4,5,6,7,8],inplace=True)

train_df['14'].replace([
'Norm',
'Feedr',
'Artery',
'PosN',
'PosA',
'RRAe',
'RRAn',
'RRNn'
],[0,1,2,3,4,5,6,7],inplace=True)
test_df['14'].replace([
'Norm',
'Feedr',
'Artery',
'PosN',
'PosA',
'RRAe',
'RRAn',
'RRNn'],[0,1,2,3,4,5,6,7],inplace=True)

train_df['15'].replace([
'1Fam',
'Duplex',
'2fmCon',
'TwnhsE',
'Twnhs'],[0,1,2,3,4],inplace=True)
test_df['15'].replace([
'1Fam',
'Duplex',
'2fmCon',
'TwnhsE',
'Twnhs'],[0,1,2,3,4],inplace=True)

train_df['16'].replace([
'1Story',
'2.5Unf',
'1.5Fin',
'2Story',
'SFoyer',
'1.5Unf',
'SLvl',
'2.5Fin'],[0,1,2,3,4,5,6,7],inplace=True)
test_df['16'].replace([
'1Story',
'2.5Unf',
'1.5Fin',
'2Story',
'SFoyer',
'1.5Unf',
'SLvl',
'2.5Fin'],[0,1,2,3,4,5,6,7],inplace=True)



train_df['21'].replace([
'Hip',
'Gable',
'Gambrel',
'Flat',
'Shed',
'Mansard'],[0,1,2,3,4,5],inplace=True)
test_df['21'].replace([
'Hip',
'Gable',
'Gambrel',
'Flat',
'Shed',
'Mansard'],[0,1,2,3,4,5],inplace=True)

train_df['22'].replace([
'CompShg',
'Tar&Grv',
'WdShngl',
'WdShake',
'ClyTile',
'Membran',
'Metal',
'Roll'
],[0,1,2,3,4,5,6,7],inplace=True)
test_df['22'].replace([
'CompShg',
'Tar&Grv',
'WdShngl',
'WdShake',
'ClyTile',
'Membran',
'Metal',
'Roll'],[0,1,2,3,4,5,6,7],inplace=True)


train_df['23'].replace([
'NA',
'MetalSd',
'Wd Sdng',
'VinylSd',
'WdShing',
'Plywood',
'HdBoard',
'BrkComm',
'BrkFace',
'AsbShng',
'CemntBd',
'Stucco',
'AsphShn',
'CBlock',
'Stone',
'ImStucc'
],[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],inplace=True)
test_df['23'].replace([
'NA',
'MetalSd',
'Wd Sdng',
'VinylSd',
'WdShing',
'Plywood',
'HdBoard',
'BrkComm',
'BrkFace',
'AsbShng',
'CemntBd',
'Stucco',
'AsphShn',
'CBlock',
'Stone',
'ImStucc'
],[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],inplace=True)

train_df['24'].replace([
'NA',
'MetalSd',
'Wd Sdng',
'VinylSd',
'Wd Shng',
'HdBoard',
'Brk Cmn',
'BrkFace',
'AsbShng',
'Plywood',
'CmentBd',
'Stucco',
'ImStucc',
'CBlock',
'AsphShn',
'Stone',
'Other'],[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],inplace=True)    
test_df['24'].replace([
'NA',
'MetalSd',
'Wd Sdng',
'VinylSd',
'Wd Shng',
'HdBoard',
'Brk Cmn',
'BrkFace',
'AsbShng',
'Plywood',
'CmentBd',
'Stucco',
'ImStucc',
'CBlock',
'AsphShn',
'Stone',
'Other'],[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],inplace=True)    

train_df['25'].replace([
'NA',
'None',
'BrkFace',
'Stone',
'BrkCmn'],[0,1,2,3,4],inplace=True)
test_df['25'].replace([
'NA',
'None',
'BrkFace',
'Stone',
'BrkCmn'],[0,1,2,3,4],inplace=True)   

train_df['27'].replace([
'TA',
'Gd',
'Ex',
'Fa'],[0,1,2,3],inplace=True)
test_df['27'].replace([
'TA',
'Gd',
'Ex',
'Fa'],[0,1,2,3],inplace=True) 

train_df['28'].replace([
'TA',
'Fa',
'Gd',
'Ex',
'Po'],[0,1,2,3,4],inplace=True)
test_df['28'].replace([
'TA',
'Fa',
'Gd',
'Ex',
'Po'],[0,1,2,3,4],inplace=True)

train_df['29'].replace([
'CBlock',
'BrkTil',
'PConc',
'Slab',
'Stone',
'Wood'],[0,1,2,3,4,5],inplace=True)
test_df['29'].replace([
'CBlock',
'BrkTil',
'PConc',
'Slab',
'Stone',
'Wood'],[0,1,2,3,4,5],inplace=True)

train_df['30'].replace([
'NA',
'TA',
'Gd',
'Fa',
'Ex'],[0,1,2,3,4],inplace=True)
test_df['30'].replace([
'NA',
'TA',
'Gd',
'Fa',
'Ex'],[0,1,2,3,4],inplace=True)

train_df['31'].replace([
'NA',
'TA',
'Fa',
'Gd',
'Po'],[0,1,2,3,4],inplace=True)
test_df['31'].replace([
'NA',
'TA',
'Fa',
'Gd',
'Po'],[0,1,2,3,4],inplace=True)

train_df['32'].replace([
'NA',
'Mn',
'No',
'Av',
'Gd'],[0,1,2,3,4],inplace=True)
test_df['32'].replace([
'NA',
'Mn',
'No',
'Av',
'Gd'],[0,1,2,3,4],inplace=True)

train_df['33'].replace([
'NA',
'BLQ',
'LwQ',
'Rec',
'Unf',
'GLQ',
'ALQ'],[0,1,2,3,4,5,6],inplace=True)
test_df['33'].replace([
'NA',
'BLQ',
'LwQ',
'Rec',
'Unf',
'GLQ',
'ALQ'],[0,1,2,3,4,5,6],inplace=True)

train_df['35'].replace([
'NA',
'Rec',
'Unf',
'ALQ',
'BLQ',
'LwQ',
'GLQ'],[0,1,2,3,4,5,6],inplace=True)
test_df['35'].replace([
'NA',
'Rec',
'Unf',
'ALQ',
'BLQ',
'LwQ',
'GLQ'],[0,1,2,3,4,5,6],inplace=True)

train_df['39'].replace([
'GasA',
'GasW',
'Wall',
'Grav',
'OthW',
'Floor'
],[0,1,2,3,4,5],inplace=True)
test_df['39'].replace([
'GasA',
'GasW',
'Wall',
'Grav',
'OthW',
'Floor'
],[0,1,2,3,4,5],inplace=True)


train_df['40'].replace([
'Gd',
'TA',
'Ex',
'Fa',
'Po'],[0,1,2,3,4],inplace=True)
test_df['40'].replace([
'Gd',
'TA',
'Ex',
'Fa',
'Po'],[0,1,2,3,4],inplace=True)

train_df['41'].replace([
'Y',
'N'],[0,1],inplace=True)
test_df['41'].replace([
'Y',
'N'],[0,1],inplace=True)


train_df['42'].replace([
'SBrkr',
'FuseA',
'FuseF',
'FuseP',
'Mix'
],[0,1,2,3,4],inplace=True)
test_df['42'].replace([
'SBrkr',
'FuseA',
'FuseF',
'FuseP',
'Mix'
],[0,1,2,3,4],inplace=True)

train_df['53'].replace([
'Gd',
'TA',
'Fa',
'Ex',
'NA'],[0,1,2,3,4],inplace=True)
test_df['53'].replace([
'Gd',
'TA',
'Fa',
'Ex',
'NA'],[0,1,2,3,4],inplace=True)

train_df['55'].replace([
'Typ',
'Sev',
'Mod',
'Maj2',
'Min1',
'Min2',
'Maj1',
'NA'],[0,1,2,3,4,5,6,7],inplace=True)
test_df['55'].replace([
'Typ',
'Sev',
'Mod',
'Maj2',
'Min1',
'Min2',
'Maj1',
'NA'],[0,1,2,3,4,5,6,7],inplace=True)

train_df['57'].replace([
'NA',
'Gd',
'TA',
'Fa',
'Po',
'Ex'],[0,1,2,3,4,5],inplace=True)
test_df['57'].replace([
'NA',
'Gd',
'TA',
'Fa',
'Po',
'Ex'],[0,1,2,3,4,5],inplace=True)

train_df['58'].replace([
'NA',
'Detchd',
'Attchd',
'Basment',
'BuiltIn',
'2Types',
'CarPort'],[0,1,2,3,4,5,6],inplace=True)
test_df['58'].replace([
'NA',
'Detchd',
'Attchd',
'Basment',
'BuiltIn',
'2Types',
'CarPort'],[0,1,2,3,4,5,6],inplace=True)

train_df['60'].replace([
'NA',
'Unf',
'Fin',
'RFn'],[0,1,2,3],inplace=True)
test_df['60'].replace([
'NA',
'Unf',
'Fin',
'RFn'],[0,1,2,3],inplace=True)

train_df['63'].replace([
'NA',
'TA',
'Fa',
'Gd',
'Po',
'Ex'],[0,1,2,3,4,5],inplace=True)
test_df['63'].replace([
'NA',
'TA',
'Fa',
'Gd',
'Po',
'Ex'],[0,1,2,3,4,5],inplace=True)
    

train_df['64'].replace([
'TA',
'NA',
'Fa',
'Gd',
'Po',
'Ex'],[0,1,2,3,4,5],inplace=True)
test_df['64'].replace([
'TA',
'NA',
'Fa',
'Gd',
'Po',
'Ex'],[0,1,2,3,4,5],inplace=True)

train_df['65'].replace([
'Y',
'N',
'P'],[0,1,2],inplace=True)
test_df['65'].replace([
'Y',
'N',
'P'],[0,1,2],inplace=True)


train_df['72'].replace([
'NA',
'Ex',
'Gd',
'Fa'],[0,1,2,3],inplace=True)
test_df['72'].replace([
'NA',
'Ex',
'Gd',
'Fa'],[0,1,2,3],inplace=True)


train_df['73'].replace([
'NA',
'MnPrv',
'GdPrv',
'GdWo',
'MnWw'],[0,1,2,3,4],inplace=True)
test_df['73'].replace([
'NA',
'MnPrv',
'GdPrv',
'GdWo',
'MnWw'],[0,1,2,3,4],inplace=True)

train_df['74'].replace([
'NA',
'Shed',
'Gar2',
'Othr',
'TenC'],[0,1,2,3,4],inplace=True)
test_df['74'].replace([
'NA',
'Shed',
'Gar2',
'Othr',
'TenC'],[0,1,2,3,4],inplace=True)

train_df['78'].replace([
'NA',
'WD',
'COD',
'New',
'ConLD',
'Oth',
'Con',
'ConLw',
'ConLI',
'CWD'],[0,1,2,3,4,5,6,7,8,9],inplace=True)
test_df['78'].replace([
'NA',
'WD',
'COD',
'New',
'ConLD',
'Oth',
'Con',
'ConLw',
'ConLI',
'CWD'],[0,1,2,3,4,5,6,7,8,9],inplace=True)

train_df['79'].replace([
'Normal',
'Abnorml',
'Alloca',
'Family',
'Partial',
'AdjLand'],[0,1,2,3,4,5],inplace=True)
test_df['79'].replace([
'Normal',
'Abnorml',
'Alloca',
'Family',
'Partial',
'AdjLand'],[0,1,2,3,4,5],inplace=True)

train_df2 = train_df.copy()
test_df2=test_df.copy()

#年份正态化
train_df['19']=(train_df['19']-train_df['19'].mean())/train_df['19'].std()
train_df['20']=(train_df['20']-train_df['20'].mean())/train_df['20'].std()


train_df['59']=(train_df['59']-train_df['59'].mean())/train_df['59'].std()
train_df['77']=(train_df['77']-train_df['77'].mean())/train_df['59'].std()
test_df['19']=(test_df['19']-test_df['19'].mean())/test_df['19'].std()
test_df['20']=(test_df['20']-test_df['20'].mean())/test_df['20'].std()
test_df['59']=(test_df['59']-test_df['59'].mean())/test_df['59'].std()
test_df['77']=(test_df['77']-test_df['77'].mean())/test_df['59'].std()

#3.3 缺失值估算 *（不知怎么填，还是用means吧）
#train_df.drop(['9'], axis=1)
#test_df.drop(['9'], axis=1)

#想遍历，不成功
for row in train_df.iterrows():
    row=row.fillna(row.mean())

train_df['1']=train_df['1'].fillna(train_df['1'].mean())
train_df['2']=train_df['2'].fillna(train_df['2'].mean())
train_df['3']=train_df['3'].fillna(train_df['3'].mean())
train_df['4']=train_df['4'].fillna(train_df['4'].mean())
train_df['5']=train_df['5'].fillna(train_df['5'].mean())
train_df['6']=train_df['6'].fillna(train_df['6'].mean())
train_df['7']=train_df['7'].fillna(train_df['7'].mean())
train_df['8']=train_df['8'].fillna(train_df['8'].mean())
train_df['9']=train_df['9'].fillna(train_df['9'].mean())
train_df['10']=train_df['10'].fillna(train_df['10'].mean())
train_df['11']=train_df['11'].fillna(train_df['11'].mean())
train_df['12']=train_df['12'].fillna(train_df['12'].mean())
train_df['13']=train_df['13'].fillna(train_df['13'].mean())
train_df['14']=train_df['14'].fillna(train_df['14'].mean())
train_df['15']=train_df['15'].fillna(train_df['15'].mean())
train_df['16']=train_df['16'].fillna(train_df['16'].mean())
train_df['17']=train_df['17'].fillna(train_df['17'].mean())
train_df['18']=train_df['18'].fillna(train_df['18'].mean())
train_df['19']=train_df['19'].fillna(train_df['19'].mean())
train_df['20']=train_df['20'].fillna(train_df['20'].mean())
train_df['21']=train_df['21'].fillna(train_df['21'].mean())
train_df['22']=train_df['22'].fillna(train_df['22'].mean())
train_df['23']=train_df['23'].fillna(train_df['23'].mean())
train_df['24']=train_df['24'].fillna(train_df['24'].mean())
train_df['25']=train_df['25'].fillna(train_df['25'].mean())
train_df['26']=train_df['26'].fillna(train_df['26'].mean())
train_df['27']=train_df['27'].fillna(train_df['27'].mean())
train_df['28']=train_df['28'].fillna(train_df['28'].mean())
train_df['29']=train_df['29'].fillna(train_df['29'].mean())
train_df['30']=train_df['30'].fillna(train_df['30'].mean())
train_df['31']=train_df['31'].fillna(train_df['31'].mean())
train_df['32']=train_df['32'].fillna(train_df['32'].mean())
train_df['33']=train_df['33'].fillna(train_df['33'].mean())
train_df['34']=train_df['34'].fillna(train_df['34'].mean())
train_df['35']=train_df['35'].fillna(train_df['35'].mean())
train_df['36']=train_df['36'].fillna(train_df['36'].mean())
train_df['37']=train_df['37'].fillna(train_df['37'].mean())
train_df['38']=train_df['38'].fillna(train_df['38'].mean())
train_df['39']=train_df['39'].fillna(train_df['39'].mean())
train_df['40']=train_df['40'].fillna(train_df['40'].mean())
train_df['41']=train_df['41'].fillna(train_df['41'].mean())
train_df['42']=train_df['42'].fillna(train_df['42'].mean())
train_df['43']=train_df['43'].fillna(train_df['43'].mean())
train_df['44']=train_df['44'].fillna(train_df['44'].mean())
train_df['45']=train_df['45'].fillna(train_df['45'].mean())
train_df['46']=train_df['46'].fillna(train_df['46'].mean())
train_df['47']=train_df['47'].fillna(train_df['47'].mean())
train_df['48']=train_df['48'].fillna(train_df['48'].mean())
train_df['49']=train_df['49'].fillna(train_df['49'].mean())
train_df['50']=train_df['50'].fillna(train_df['50'].mean())
train_df['51']=train_df['51'].fillna(train_df['51'].mean())
train_df['52']=train_df['52'].fillna(train_df['52'].mean())
train_df['53']=train_df['53'].fillna(train_df['53'].mean())
train_df['54']=train_df['54'].fillna(train_df['54'].mean())
train_df['55']=train_df['55'].fillna(train_df['55'].mean())
train_df['56']=train_df['56'].fillna(train_df['56'].mean())
train_df['57']=train_df['57'].fillna(train_df['57'].mean())
train_df['58']=train_df['58'].fillna(train_df['58'].mean())
train_df['59']=train_df['59'].fillna(train_df['59'].mean())
train_df['60']=train_df['60'].fillna(train_df['60'].mean())
train_df['61']=train_df['61'].fillna(train_df['61'].mean())
train_df['62']=train_df['62'].fillna(train_df['62'].mean())
train_df['63']=train_df['63'].fillna(train_df['63'].mean())
train_df['64']=train_df['64'].fillna(train_df['64'].mean())
train_df['65']=train_df['65'].fillna(train_df['65'].mean())
train_df['66']=train_df['66'].fillna(train_df['66'].mean())
train_df['67']=train_df['67'].fillna(train_df['67'].mean())
train_df['68']=train_df['68'].fillna(train_df['68'].mean())
train_df['69']=train_df['69'].fillna(train_df['69'].mean())
train_df['70']=train_df['70'].fillna(train_df['70'].mean())
train_df['71']=train_df['71'].fillna(train_df['71'].mean())
train_df['72']=train_df['72'].fillna(train_df['72'].mean())
train_df['73']=train_df['73'].fillna(train_df['73'].mean())
train_df['74']=train_df['74'].fillna(train_df['74'].mean())
train_df['75']=train_df['75'].fillna(train_df['75'].mean())
train_df['76']=train_df['76'].fillna(train_df['76'].mean())
train_df['77']=train_df['77'].fillna(train_df['77'].mean())
train_df['78']=train_df['78'].fillna(train_df['78'].mean())
train_df['79']=train_df['79'].fillna(train_df['79'].mean())
train_df['SP']=train_df['SP'].fillna(train_df['SP'].mean())

test_df['1']=test_df['1'].fillna(test_df['1'].mean())
test_df['2']=test_df['2'].fillna(test_df['2'].mean())
test_df['3']=test_df['3'].fillna(test_df['3'].mean())
test_df['4']=test_df['4'].fillna(test_df['4'].mean())
test_df['5']=test_df['5'].fillna(test_df['5'].mean())
test_df['6']=test_df['6'].fillna(test_df['6'].mean())
test_df['7']=test_df['7'].fillna(test_df['7'].mean())
test_df['8']=test_df['8'].fillna(test_df['8'].mean())
train_df['9']=train_df['9'].fillna(train_df['9'].mean())
test_df['10']=test_df['10'].fillna(test_df['10'].mean())
test_df['11']=test_df['11'].fillna(test_df['11'].mean())
test_df['12']=test_df['12'].fillna(test_df['12'].mean())
test_df['13']=test_df['13'].fillna(test_df['13'].mean())
test_df['14']=test_df['14'].fillna(test_df['14'].mean())
test_df['15']=test_df['15'].fillna(test_df['15'].mean())
test_df['16']=test_df['16'].fillna(test_df['16'].mean())
test_df['17']=test_df['17'].fillna(test_df['17'].mean())
test_df['18']=test_df['18'].fillna(test_df['18'].mean())
test_df['19']=test_df['19'].fillna(test_df['19'].mean())
test_df['20']=test_df['20'].fillna(test_df['20'].mean())
test_df['21']=test_df['21'].fillna(test_df['21'].mean())
test_df['22']=test_df['22'].fillna(test_df['22'].mean())
test_df['23']=test_df['23'].fillna(test_df['23'].mean())
test_df['24']=test_df['24'].fillna(test_df['24'].mean())
test_df['25']=test_df['25'].fillna(test_df['25'].mean())
test_df['26']=test_df['26'].fillna(test_df['26'].mean())
test_df['27']=test_df['27'].fillna(test_df['27'].mean())
test_df['28']=test_df['28'].fillna(test_df['28'].mean())
test_df['29']=test_df['29'].fillna(test_df['29'].mean())
test_df['30']=test_df['30'].fillna(test_df['30'].mean())
test_df['31']=test_df['31'].fillna(test_df['31'].mean())
test_df['32']=test_df['32'].fillna(test_df['32'].mean())
test_df['33']=test_df['33'].fillna(test_df['33'].mean())
test_df['34']=test_df['34'].fillna(test_df['34'].mean())
test_df['35']=test_df['35'].fillna(test_df['35'].mean())
test_df['36']=test_df['36'].fillna(test_df['36'].mean())
test_df['37']=test_df['37'].fillna(test_df['37'].mean())
test_df['38']=test_df['38'].fillna(test_df['38'].mean())
test_df['39']=test_df['39'].fillna(test_df['39'].mean())
test_df['40']=test_df['40'].fillna(test_df['40'].mean())
test_df['41']=test_df['41'].fillna(test_df['41'].mean())
test_df['42']=test_df['42'].fillna(test_df['42'].mean())
test_df['43']=test_df['43'].fillna(test_df['43'].mean())
test_df['44']=test_df['44'].fillna(test_df['44'].mean())
test_df['45']=test_df['45'].fillna(test_df['45'].mean())
test_df['46']=test_df['46'].fillna(test_df['46'].mean())
test_df['47']=test_df['47'].fillna(test_df['47'].mean())
test_df['48']=test_df['48'].fillna(test_df['48'].mean())
test_df['49']=test_df['49'].fillna(test_df['49'].mean())
test_df['50']=test_df['50'].fillna(test_df['50'].mean())
test_df['51']=test_df['51'].fillna(test_df['51'].mean())
test_df['52']=test_df['52'].fillna(test_df['52'].mean())
test_df['53']=test_df['53'].fillna(test_df['53'].mean())
test_df['54']=test_df['54'].fillna(test_df['54'].mean())
test_df['55']=test_df['55'].fillna(test_df['55'].mean())
test_df['56']=test_df['56'].fillna(test_df['56'].mean())
test_df['57']=test_df['57'].fillna(test_df['57'].mean())
test_df['58']=test_df['58'].fillna(test_df['58'].mean())
test_df['59']=test_df['59'].fillna(test_df['59'].mean())
test_df['60']=test_df['60'].fillna(test_df['60'].mean())
test_df['61']=test_df['61'].fillna(test_df['61'].mean())
test_df['62']=test_df['62'].fillna(test_df['62'].mean())
test_df['63']=test_df['63'].fillna(test_df['63'].mean())
test_df['64']=test_df['64'].fillna(test_df['64'].mean())
test_df['65']=test_df['65'].fillna(test_df['65'].mean())
test_df['66']=test_df['66'].fillna(test_df['66'].mean())
test_df['67']=test_df['67'].fillna(test_df['67'].mean())
test_df['68']=test_df['68'].fillna(test_df['68'].mean())
test_df['69']=test_df['69'].fillna(test_df['69'].mean())
test_df['70']=test_df['70'].fillna(test_df['70'].mean())
test_df['71']=test_df['71'].fillna(test_df['71'].mean())
test_df['72']=test_df['72'].fillna(test_df['72'].mean())
test_df['73']=test_df['73'].fillna(test_df['73'].mean())
test_df['74']=test_df['74'].fillna(test_df['74'].mean())
test_df['75']=test_df['75'].fillna(test_df['75'].mean())
test_df['76']=test_df['76'].fillna(test_df['76'].mean())
test_df['77']=test_df['77'].fillna(test_df['77'].mean())
test_df['78']=test_df['78'].fillna(test_df['78'].mean())
test_df['79']=test_df['79'].fillna(test_df['79'].mean())
#test_df['SP']=test_df['SP'].fillna(test_df['SP'].mean())

filename1='train2.csv'
filename2='test2.csv'
train_df.to_csv(filename1,index=None,encoding='utf-8')#注意文件打开是没法写入的，注意要输出的DataFrame是df
test_df.to_csv(filename2,index=None,encoding='utf-8')

数据处理第二部分

# 3.4 清除异常值，缩放 
#定义异常值
def outer(df, column): 
    df_ = df.copy() #不在原来数据修改
    # 这里将大于上四分位数(Q3)的设定为异常值
    df_['isOutlier'] = df_[column] > df_[column].quantile(0.95)
    df_rst = df_[df_['isOutlier'] == True]
    print(df_rst)

    
#想遍历，不成功
for row in train_df.iterrows():
    outer=outer(train_df,row)
    train_df.drop(outer)
    

train_df = pd.read_csv("C:/Users/sl6118919/Python_Learn/超思机器学习/kaggle/hourse/train2.csv")
test_df = pd.read_csv("C:/Users/sl6118919/Python_Learn/超思机器学习/kaggle/hourse/test2.csv")

#RobustScaler 稳健缩放 (抗异常值缩放) ####缩放的行列没弄明白，暂时不做
#import numpy as np
#from sklearn.preprocessing import RobustScaler
#train_df=np.array(train_df)
#test_df=np.array(test_df)
#model = RobustScaler(with_centering = True, with_scaling = True, 
                    quantile_range = (0, 95.0))
# with_centering = True => 中心归零，变量X将会变为：X - X.median()
# with_centering = True => 数值标准化，变量X将会除以变量分位数区间（区间由用户设定）
#model.fit(train_df.reshape(-1,1)) # 在训练集上训练
# 转换缩放训练集与测试集
#train_df  = model.transform(train_df.reshape(-1,1)).reshape(-1) # 转换训练集
#test_df  = model.transform(test_df.reshape(-1,1)).reshape(-1) # 转换测试集
#train_df  = pd.DataFrame(train_df )  
#test_df   = pd.DataFrame(test_df  )  

#热力图看不清
sns.heatmap(train_df.corr(), annot=True).set_title("Corelation of attributes")
fig=plt.gcf()
fig.set_size_inches(15,15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

import numpy as np
from scipy.stats import pearsonr
from sklearn.feature_selection import SelectKBest


train_y=np.array(train_df['SP'])
train_df2=train_df.drop(['Id'], axis=1)
train_ar=np.array(train_df2)
#train_df.drop(['9'], axis=1)

for idx in range(train_ar.shape[1]):
    pea_score, p_value = pearsonr(train_ar[:,idx],train_y)
    #print(pea_score)
    #print(p_value)
    print(f"第{idx + 1}个变量和目标的皮尔森相关系数的绝对值为{round(np.abs(pea_score),2)}, p-值为{round(p_value,3)}")

模型训练

#构建X y
# 检测数据中是否存在NaN,如果存在就返回True
y_train=train_df['SP']
print(np.any(y_train.isnull())==True)
#y_train=y_train.values.tolist()
X_train=train_df[['46',
'61',
'62',
'38',
'43',
'30',
'27',
'49',
'54',
'19',
'20'
]]
X_test=test_df[['46',
'61',
'62',
'38',
'43',
'30',
'27',
'49',
'54',
'19',
'20'
]]
print(np.any(X_train.isnull())==True)
#X=X.values.tolist()


5.交叉验证

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm
from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.model_selection import cross_val_score #score evaluation
from sklearn.model_selection import cross_val_predict #prediction
kfold = KFold(n_splits=10, random_state=22) # k=10, split the data into 10 equal parts
xyz=[]
accuracy=[]
std=[]
classifiers=['Linear Svm',
             'Radial Svm',
             'Logistic Regression',
             'KNN',
             'Decision Tree',
             'Naive Bayes',
             'Random Forest']
models=[svm.SVC(kernel='linear'),
        svm.SVC(kernel='rbf'),
        LogisticRegression(),
        KNeighborsClassifier(n_neighbors=9),
        DecisionTreeClassifier(),
        GaussianNB(),
        RandomForestClassifier(n_estimators=100)]




for i in models:
    model = i
    cv_result = cross_val_score(model,X_train,y_train, cv = kfold,scoring = "neg_root_mean_squared_error")
    cv_result=cv_result
    xyz.append(cv_result.mean())
    std.append(cv_result.std())
    accuracy.append(cv_result)
new_models_dataframe2=pd.DataFrame({'CV Mean':xyz,'Std':std},index=classifiers)       
new_models_dataframe2

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

6.产生结果提交，总结

#写入
filename='house_submission.csv'
df1=test_df['Id']
df2 = pd.DataFrame(y_pred, columns=['SalePrice'])
df=pd.concat([df1, df2],axis=1)
df.to_csv(filename,index=None,encoding='utf-8')#
