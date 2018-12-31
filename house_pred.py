# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 18:28:56 2018

@author: rohanh
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import Imputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
%matplotlib inline
house_df = pd.read_csv('house_price_train.csv',index_col=0)
#summary = house_df.describe(include='all')
#typ = house_df.dtypes
house_df['MSSubClass'] = house_df['MSSubClass'].astype('object')
house_df['OverallQual'] = house_df['OverallQual'].astype('object')
house_df['OverallCond'] = house_df['OverallCond'].astype('object')
#summary = house_df.describe(include='all')
### Fields Alley, PoolQC, Fence and MiscFeature has very less count compare to total count
### which means there are too many missing values. Hence we will forst check the impact
### of these fields on the target variable and consider them only if they are contributing.
### Find impact of Alley, PoolQC, Fence and MiscFeature variables on SalePrice 
stats.linregress(house_df['Alley'].astype('category').cat.codes,house_df['SalePrice'])
stats.linregress(house_df['PoolQC'].astype('category').cat.codes,house_df['SalePrice'])
stats.linregress(house_df['Fence'].astype('category').cat.codes,house_df['SalePrice'])
stats.linregress(house_df['MiscFeature'].astype('category').cat.codes,house_df['SalePrice'])

###Fence is not contributing to the SalePrice
#### Create dummy variables to store these catagorical values
house_df['Alley_dummy'] = house_df['Alley'].apply(lambda x: 0 if pd.isnull(x) else 1).astype('object')
house_df['PoolQC_dummy'] = house_df['PoolQC'].apply(lambda x: 0 if pd.isnull(x) else 1).astype('object')
house_df['MiscFeature_dummy'] = house_df['MiscFeature'].apply(lambda x: 0 if pd.isnull(x) else 1).astype('object')

### Check summary again

summary = house_df.describe(include='all')


#### Fill NaN (missing values) using Interpolate method
hdf = house_df.apply(lambda x : x.astype('category').cat.codes.replace(-1, np.nan).interpolate(method='linear', limit_direction='both').astype(int).astype('category').cat.rename_categories(x.astype('category').cat.categories) if x.dtype=='object' else x.interpolate())


#### Fill NaN (missing values) using inputer function of sklearn for numeric values
#imp = Imputer(missing_values='NaN', strategy='median', axis=0)
#Input_house_df =pd.DataFrame(house_df, columns=house_df.select_dtypes(include='number').columns)
#imp.fit(Input_house_df)
#Input_house_df = imp.transform(Input_house_df)
#Input_house_df =pd.DataFrame(Input_house_df, columns=house_df.select_dtypes(include='number').columns).set_index(house_df.index)
#house_df.loc[0:100][['LotArea','SalePrice']].plot.area()

#corr_matrix = hdf.corr()

#nas = hdf.isna().sum()
#### Outlier Detection and removal of numeric/continuous fields
#Field: LotFrontage
hdf[['LotFrontage','SalePrice']].corr().loc['LotFrontage','SalePrice']
sns.boxplot(hdf['LotFrontage'])
hdf['LotFrontage'].quantile([0,0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.95, 1])
l_replace = hdf['LotFrontage'].quantile(0.05,interpolation='nearest')
h_replace = hdf['LotFrontage'].quantile(0.95,interpolation='nearest')
hdf['LotFrontage'] = hdf['LotFrontage'].apply(lambda x: l_replace if x<l_replace else x)
hdf['LotFrontage'] = hdf['LotFrontage'].apply(lambda x: h_replace if x>h_replace else x)
sns.boxplot(hdf['LotFrontage'])

#Field: LotArea
hdf[['LotArea','SalePrice']].corr().loc['LotArea','SalePrice']
sns.boxplot(hdf['LotArea'])
hdf['LotArea'].quantile([0,0.02, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.95, 1])
l_replace = hdf['LotArea'].quantile(0.02,interpolation='nearest')
h_replace = hdf['LotArea'].quantile(0.95,interpolation='nearest')
hdf['LotArea'] = hdf['LotArea'].apply(lambda x: l_replace if x<l_replace else x)
hdf['LotArea'] = hdf['LotArea'].apply(lambda x: h_replace if x>h_replace else x)
sns.boxplot(hdf['LotArea'])

#Field: YearBuilt
hdf[['YearBuilt','SalePrice']].corr().loc['YearBuilt','SalePrice']
sns.boxplot(hdf['YearBuilt'])
hdf['YearBuilt'].quantile([0,0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.95, 1])
l_replace = hdf['YearBuilt'].quantile(0.01,interpolation='nearest')
#h_replace = hdf['YearBuilt'].quantile(0.95,interpolation='nearest')
hdf['YearBuilt'] = hdf['YearBuilt'].apply(lambda x: l_replace if x<l_replace else x)
#hdf['YearBuilt'] = hdf['YearBuilt'].apply(lambda x: h_replace if x>h_replace else x)
sns.boxplot(hdf['YearBuilt'])

#Field: YearRemodAdd
hdf[['YearRemodAdd','SalePrice']].corr().loc['YearRemodAdd','SalePrice']
sns.boxplot(hdf['YearRemodAdd'])
#hdf['YearRemodAdd'].quantile([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.95, 1])
#l_replace = hdf['YearRemodAdd'].quantile(0.01,interpolation='nearest')
#h_replace = hdf['YearRemodAdd'].quantile(0.95,interpolation='nearest')
#hdf['YearRemodAdd'] = hdf['YearRemodAdd'].apply(lambda x: l_replace if x<l_replace else x)
#hdf['YearRemodAdd'] = hdf['YearRemodAdd'].apply(lambda x: h_replace if x>h_replace else x)
sns.boxplot(hdf['YearRemodAdd'])

#Field: MasVnrArea
hdf[['MasVnrArea','SalePrice']].corr().loc['MasVnrArea','SalePrice']
sns.boxplot(hdf['MasVnrArea'])
hdf['MasVnrType'].value_counts()
### The MasVnrType and MasVnrArea are correlated 
### also these fields have majority of 'none' values means missing Masonry veneer
### First we will check if MasVnrType has impact on SalePrice and if yes, 
### we will combine these fields into a new dumy field as MasVnr_dummy
stats.linregress(house_df['MasVnrType'].astype('category').cat.codes,house_df['SalePrice'])
### Pvalue is 0.98 hence we will not create the new variable
#hdf['MasVnr_dummy'] = hdf['MasVnrType'].apply(lambda x: 0 if x=='None' else 1).astype('object')

#Field : BsmtFinSF1
hdf[['BsmtFinSF1','SalePrice']].corr().loc['BsmtFinSF1','SalePrice']
sns.boxplot(hdf['BsmtFinSF1'])
hdf['BsmtFinSF1'].quantile([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.95, 1])
hdf['BsmtFinSF1'].value_counts()
#l_replace = hdf['YearRemodAdd'].quantile(0.01,interpolation='nearest')
h_replace = hdf['BsmtFinSF1'].quantile(0.95,interpolation='nearest')
#hdf['YearRemodAdd'] = hdf['YearRemodAdd'].apply(lambda x: l_replace if x<l_replace else x)
hdf['BsmtFinSF1'] = hdf['BsmtFinSF1'].apply(lambda x: h_replace if x>h_replace else x)
sns.boxplot(hdf['BsmtFinSF1'])


#Field : BsmtFinSF2
hdf[['BsmtFinSF2','SalePrice']].corr().loc['BsmtFinSF2','SalePrice']
sns.boxplot(hdf['BsmtFinSF2'])
### The box plot contains too manu outliers and the median is almost zero ~ min value
### this shows that the distribution is highly skewed. We will first check if the 
### second basement type affects the SalePrice and if yes we will create the new dummy variable
stats.linregress(house_df['BsmtFinType2'].astype('category').cat.codes,house_df['SalePrice'])
### Pvalue 5.28 hence we will discard this field from predictors

#Field: BsmtUnfSF
hdf[['BsmtUnfSF','SalePrice']].corr().loc['BsmtUnfSF','SalePrice']
sns.boxplot(hdf['BsmtUnfSF'])
hdf['BsmtUnfSF'].quantile([0,0.02, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.97, 1])
#l_replace = hdf['BsmtUnfSF'].quantile(0.02,interpolation='nearest')
h_replace = hdf['BsmtUnfSF'].quantile(0.97,interpolation='nearest')
#hdf['BsmtUnfSF'] = hdf['BsmtUnfSF'].apply(lambda x: l_replace if x<l_replace else x)
hdf['BsmtUnfSF'] = hdf['BsmtUnfSF'].apply(lambda x: h_replace if x>h_replace else x)
sns.boxplot(hdf['BsmtUnfSF'])

#Field: TotalBsmtSF
hdf[['TotalBsmtSF','SalePrice']].corr().loc['TotalBsmtSF','SalePrice']
sns.boxplot(hdf['TotalBsmtSF'])
hdf['TotalBsmtSF'].quantile([0,0.03, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.98, 1])
l_replace = hdf['TotalBsmtSF'].quantile(0.03,interpolation='nearest')
h_replace = hdf['TotalBsmtSF'].quantile(0.98,interpolation='nearest')
hdf['TotalBsmtSF'] = hdf['TotalBsmtSF'].apply(lambda x: l_replace if x<l_replace else x)
hdf['TotalBsmtSF'] = hdf['TotalBsmtSF'].apply(lambda x: h_replace if x>h_replace else x)
sns.boxplot(hdf['TotalBsmtSF'])

#Field: 1stFlrSF
hdf[['1stFlrSF','SalePrice']].corr().loc['1stFlrSF','SalePrice']
sns.boxplot(hdf['1stFlrSF'])
hdf['1stFlrSF'].quantile([0,0.03, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.98, 1])
#l_replace = hdf['1stFlrSF'].quantile(0.03,interpolation='nearest')
h_replace = hdf['1stFlrSF'].quantile(0.98,interpolation='nearest')
#hdf['1stFlrSF'] = hdf['1stFlrSF'].apply(lambda x: l_replace if x<l_replace else x)
hdf['1stFlrSF'] = hdf['1stFlrSF'].apply(lambda x: h_replace if x>h_replace else x)
sns.boxplot(hdf['1stFlrSF'])

#Field: 2ndFlrSF
hdf[['2ndFlrSF','SalePrice']].corr().loc['2ndFlrSF','SalePrice']
sns.boxplot(hdf['2ndFlrSF'])
hdf['2ndFlrSF'].quantile([0,0.03, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.98, 1])
#l_replace = hdf['2ndFlrSF'].quantile(0.03,interpolation='nearest')
h_replace = hdf['2ndFlrSF'].quantile(0.98,interpolation='nearest')
#hdf['2ndFlrSF'] = hdf['2ndFlrSF'].apply(lambda x: l_replace if x<l_replace else x)
hdf['2ndFlrSF'] = hdf['2ndFlrSF'].apply(lambda x: h_replace if x>h_replace else x)
sns.boxplot(hdf['2ndFlrSF'])


#Field: LowQualFinSF
hdf[['LowQualFinSF','SalePrice']].corr().loc['LowQualFinSF','SalePrice']
sns.boxplot(hdf['LowQualFinSF'])
#hdf['LowQualFinSF_dummy'] = hdf['LowQualFinSF'].apply(lambda x: 0 if x==0 else 1).astype('category')
stats.linregress(hdf['LowQualFinSF'].apply(lambda x: 0 if x==0 else 1).astype('category').cat.codes,house_df['SalePrice'])
### Pvalue is 0.06 not <0.05 hence we will not consider this

#Field: GrLivArea
hdf[['GrLivArea','SalePrice']].corr().loc['GrLivArea','SalePrice']
sns.boxplot(hdf['GrLivArea'])
hdf['GrLivArea'].quantile([0,0.03, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.97, 1])
#l_replace = hdf['GrLivArea'].quantile(0.03,interpolation='nearest')
h_replace = hdf['GrLivArea'].quantile(0.97,interpolation='nearest')
#hdf['GrLivArea'] = hdf['GrLivArea'].apply(lambda x: l_replace if x<l_replace else x)
hdf['GrLivArea'] = hdf['GrLivArea'].apply(lambda x: h_replace if x>h_replace else x)
sns.boxplot(hdf['GrLivArea'])

#Field: BsmtFullBath
hdf[['BsmtFullBath','SalePrice']].corr().loc['BsmtFullBath','SalePrice']
sns.boxplot(hdf['BsmtFullBath'])
hdf['BsmtFullBath'].value_counts()
hdf['BsmtFullBath'] = hdf['BsmtFullBath'].astype('category')
stats.linregress(hdf['BsmtFullBath'].astype('category').cat.codes,house_df['SalePrice'])
### Pvalue = 1.55

#Field: BsmtHalfBath
hdf[['BsmtHalfBath','SalePrice']].corr().loc['BsmtHalfBath','SalePrice']
sns.boxplot(hdf['BsmtHalfBath'])
hdf['BsmtHalfBath'].value_counts()
hdf['BsmtHalfBath'] = hdf['BsmtHalfBath'].astype('category')
stats.linregress(hdf['BsmtHalfBath'].astype('category').cat.codes,house_df['SalePrice'])
### Pvalue 0.52

#Field: FullBath
hdf[['FullBath','SalePrice']].corr().loc['FullBath','SalePrice']
sns.boxplot(hdf['FullBath'])
hdf['FullBath'].value_counts()
stats.linregress(hdf['FullBath'].astype('category').cat.codes,house_df['SalePrice'])
### pvalue = 1.23

#Field: HalfBath
hdf[['HalfBath','SalePrice']].corr().loc['HalfBath','SalePrice']
sns.boxplot(hdf['HalfBath'])
hdf['HalfBath'].value_counts()
stats.linregress(hdf['HalfBath'].astype('category').cat.codes,house_df['SalePrice'])
### Pvalue = 1.65

#Field: BedroomAbvGr
hdf[['BedroomAbvGr','SalePrice']].corr().loc['BedroomAbvGr','SalePrice']
sns.boxplot(hdf['BedroomAbvGr'])
hdf['BedroomAbvGr'].quantile([0,0.02, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.95, 1])
l_replace = hdf['BedroomAbvGr'].quantile(0.03,interpolation='nearest')
h_replace = hdf['BedroomAbvGr'].quantile(0.95,interpolation='nearest')
hdf['BedroomAbvGr'] = hdf['BedroomAbvGr'].apply(lambda x: l_replace if x<l_replace else x)
hdf['BedroomAbvGr'] = hdf['BedroomAbvGr'].apply(lambda x: h_replace if x>h_replace else x)
sns.boxplot(hdf['BedroomAbvGr'])

#Field: KitchenAbvGr
hdf[['KitchenAbvGr','SalePrice']].corr().loc['KitchenAbvGr','SalePrice']
sns.boxplot(hdf['KitchenAbvGr'])
hdf['KitchenAbvGr'].value_counts()
stats.linregress(hdf['KitchenAbvGr'].astype('category').cat.codes,house_df['SalePrice'])
### Pvalue = 1.86

#Field: TotRmsAbvGrd
hdf[['TotRmsAbvGrd','SalePrice']].corr().loc['TotRmsAbvGrd','SalePrice']
sns.boxplot(hdf['TotRmsAbvGrd'])
hdf['TotRmsAbvGrd'].quantile([0,0.02, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.95, 1])
#l_replace = hdf['TotRmsAbvGrd'].quantile(0.03,interpolation='nearest')
h_replace = hdf['TotRmsAbvGrd'].quantile(0.95,interpolation='nearest')
#hdf['TotRmsAbvGrd'] = hdf['TotRmsAbvGrd'].apply(lambda x: l_replace if x<l_replace else x)
hdf['TotRmsAbvGrd'] = hdf['TotRmsAbvGrd'].apply(lambda x: h_replace if x>h_replace else x)
sns.boxplot(hdf['TotRmsAbvGrd'])


#Field: Fireplaces
hdf[['Fireplaces','SalePrice']].corr().loc['Fireplaces','SalePrice']
sns.boxplot(hdf['Fireplaces'])
hdf['Fireplaces'].quantile([0,0.02, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.95, 1])
#l_replace = hdf['TotRmsAbvGrd'].quantile(0.03,interpolation='nearest')
h_replace = hdf['Fireplaces'].quantile(0.95,interpolation='nearest')
#hdf['TotRmsAbvGrd'] = hdf['TotRmsAbvGrd'].apply(lambda x: l_replace if x<l_replace else x)
hdf['Fireplaces'] = hdf['Fireplaces'].apply(lambda x: h_replace if x>h_replace else x)
sns.boxplot(hdf['Fireplaces'])

#Field: GarageYrBlt
hdf[['GarageYrBlt','SalePrice']].corr().loc['GarageYrBlt','SalePrice']
sns.boxplot(hdf['GarageYrBlt'])
hdf['GarageYrBlt'].quantile([0,0.02, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.95, 1])
l_replace = hdf['GarageYrBlt'].quantile(0.02,interpolation='nearest')
#h_replace = hdf['Fireplaces'].quantile(0.95,interpolation='nearest')
hdf['GarageYrBlt'] = hdf['GarageYrBlt'].apply(lambda x: l_replace if x<l_replace else x)
#hdf['Fireplaces'] = hdf['Fireplaces'].apply(lambda x: h_replace if x>h_replace else x)
sns.boxplot(hdf['GarageYrBlt'])

#Field: GarageCars
hdf[['GarageCars','SalePrice']].corr().loc['GarageCars','SalePrice']
sns.boxplot(hdf['GarageCars'])
hdf['GarageCars'].quantile([0,0.02, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.95, 1])
#l_replace = hdf['GarageYrBlt'].quantile(0.02,interpolation='nearest')
h_replace = hdf['GarageCars'].quantile(0.95,interpolation='nearest')
#hdf['GarageYrBlt'] = hdf['GarageYrBlt'].apply(lambda x: l_replace if x<l_replace else x)
hdf['GarageCars'] = hdf['GarageCars'].apply(lambda x: h_replace if x>h_replace else x)
sns.boxplot(hdf['GarageCars'])

#Field: GarageArea
hdf[['GarageArea','SalePrice']].corr().loc['GarageArea','SalePrice']
sns.boxplot(hdf['GarageArea'])
hdf['GarageArea'].quantile([0,0.02, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.97, 1])
#l_replace = hdf['GarageYrBlt'].quantile(0.02,interpolation='nearest')
h_replace = hdf['GarageArea'].quantile(0.97,interpolation='nearest')
#hdf['GarageYrBlt'] = hdf['GarageYrBlt'].apply(lambda x: l_replace if x<l_replace else x)
hdf['GarageArea'] = hdf['GarageArea'].apply(lambda x: h_replace if x>h_replace else x)
sns.boxplot(hdf['GarageArea'])


#Field: WoodDeckSF
hdf[['WoodDeckSF','SalePrice']].corr().loc['WoodDeckSF','SalePrice']
sns.boxplot(hdf['WoodDeckSF'])
hdf['WoodDeckSF'].quantile([0,0.02, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.97, 1])
#l_replace = hdf['GarageYrBlt'].quantile(0.02,interpolation='nearest')
h_replace = hdf['WoodDeckSF'].quantile(0.97,interpolation='nearest')
#hdf['GarageYrBlt'] = hdf['GarageYrBlt'].apply(lambda x: l_replace if x<l_replace else x)
hdf['WoodDeckSF'] = hdf['WoodDeckSF'].apply(lambda x: h_replace if x>h_replace else x)
sns.boxplot(hdf['WoodDeckSF'])

#Field: OpenPorchSF
hdf[['OpenPorchSF','SalePrice']].corr().loc['OpenPorchSF','SalePrice']
sns.boxplot(hdf['OpenPorchSF'])
hdf['OpenPorchSF'].quantile([0,0.02, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.94, 1])
#l_replace = hdf['OpenPorchSF'].quantile(0.02,interpolation='nearest')
h_replace = hdf['OpenPorchSF'].quantile(0.94,interpolation='nearest')
#hdf['OpenPorchSF'] = hdf['OpenPorchSF'].apply(lambda x: l_replace if x<l_replace else x)
hdf['OpenPorchSF'] = hdf['OpenPorchSF'].apply(lambda x: h_replace if x>h_replace else x)
sns.boxplot(hdf['OpenPorchSF'])

#Field: EnclosedPorch
hdf[['EnclosedPorch','SalePrice']].corr().loc['EnclosedPorch','SalePrice']
sns.boxplot(hdf['EnclosedPorch'])
hdf['EnclosedPorch'].value_counts()
stats.linregress(hdf['EnclosedPorch'].astype('category').cat.codes,house_df['SalePrice'])
### Pvalue = 1.54

#Field: 3SsnPorch
hdf[['3SsnPorch','SalePrice']].corr().loc['3SsnPorch','SalePrice']
sns.boxplot(hdf['3SsnPorch'])
stats.linregress(hdf['3SsnPorch'].astype('category').cat.codes,house_df['SalePrice'])
### Pvalue = 0.07

#Field: ScreenPorch
hdf[['ScreenPorch','SalePrice']].corr().loc['ScreenPorch','SalePrice']
sns.boxplot(hdf['ScreenPorch'])
stats.linregress(hdf['ScreenPorch'].astype('category').cat.codes,house_df['SalePrice'])
### Pvalue = 1.25

#Field: PoolArea
hdf[['PoolArea','SalePrice']].corr().loc['PoolArea','SalePrice']
sns.boxplot(hdf['PoolArea'])
hdf['PoolArea'].value_counts()
hdf['PoolArea_dummy']=hdf['PoolArea'].apply(lambda x: 0 if x==0 else 1).astype('category')
stats.linregress(hdf['PoolArea_dummy'].astype('category').cat.codes,house_df['SalePrice'])

### Pvalue = 0.0007

#Field: MiscVal
hdf[['MiscVal','SalePrice']].corr().loc['MiscVal','SalePrice']
sns.boxplot(hdf['MiscVal'])
stats.linregress(hdf['MiscVal'].astype('category').cat.codes,house_df['SalePrice'])
### Pvalue = 0.10

#Field: MoSold
hdf[['MoSold','SalePrice']].corr().loc['MoSold','SalePrice']
sns.boxplot(hdf['MoSold'])


#Field: YrSold
hdf[['YrSold','SalePrice']].corr().loc['YrSold','SalePrice']
sns.boxplot(hdf['YrSold'])


summary = hdf.describe(include='all')

typ = hdf.dtypes

### changing the categorical fields to numbers
hdf['Exterior2nd'].value_counts()
dict_MSZoning = {'RL':1, 'RM':2, 'FV':3,'RH':4, 'C (all)':5}
dict_Street = {'Pave':1,'Grvl':2}
dict_LotShape = {'Reg':1, 'IR1':2, 'IR2':3,'IR3':4}
dict_LandContour = {'Lvl':1, 'Bnk':2, 'HLS':3,'Low':4}
dict_Utilities = {'AllPub':1,'NoSeWa':2}
dict_LotConfig = {'Inside':1, 'Corner':2, 'CulDSac':3,'FR2':4, 'FR3':5}
dict_LandSlope = {'Gtl':1, 'Mod':2, 'Sev':3}
dict_Neighborhood = {'NAmes':1, 'CollgCr':2, 'OldTown':3,'Edwards':4, 'Somerst':5,
                     'Gilbert':6,'NridgHt':7,'Sawyer':8,'NWAmes':9,'SawyerW':10,
                     'BrkSide':11,'Crawfor':12,'Mitchel':13,'NoRidge':14,'Timber':15,
                     'IDOTRR':16,'ClearCr':17,'SWISU':18,'StoneBr':19,'Blmngtn':20,
                     'MeadowV':21,'BrDale':22,'Veenker':23,'NPkVill':24,'Blueste':25}
dict_Condition1={'Norm':1,'Feedr':2,'Artery':3,'RRAn':4,'PosN':5,
                 'RRAe':6,'PosA':7,'RRNn':8,'RRNe':9}
dict_Condition2={'Norm':1,'Feedr':2,'Artery':3,'RRAn':4,'PosN':5,
                 'RRAe':6,'PosA':7,'RRNn':8}
dict_BldgType={'1Fam':1,'TwnhsE':2,'Duplex':3,'Twnhs':4,'2fmCon':5}
dict_HouseStyle={'1Story':1,'2Story':2,'1.5Fin':3,'SLvl':4,'SFoyer':5,
                 '1.5Unf':6,'2.5Unf':7,'2.5Fin':8}
dict_RoofStyle={'Gable':1,'Hip':2,'Flat':3,'Gambrel':4,'Mansard':5,'Shed':6}
dict_RoofMatl={'CompShg':1,'Tar&Grv':2,'WdShngl':3,'WdShake':4,'Roll':5,
                 'Metal':6,'Membran':7,'ClyTile':8}
dict_Exterior1st = {'VinylSd':1, 'HdBoard':2, 'MetalSd':3,'Wd Sdng':4, 'Plywood':5,
                     'CemntBd':6,'BrkFace':7,'WdShing':8,'Stucco':9,'AsbShng':10,
                     'Stone':11,'BrkComm':12,'ImStucc':13,'CBlock':14,'AsphShn':15}
dict_Exterior2nd = {'VinylSd':1, 'HdBoard':2, 'MetalSd':3,'Wd Sdng':4, 'Plywood':5,
                     'CmentBd':6,'BrkFace':7,'Wd Shng':8,'Stucco':9,'AsbShng':10,
                     'Stone':11,'Brk Cmn':12,'ImStucc':13,'CBlock':14,'AsphShn':15,
                     'Other':16}
dict_ExterQual={'TA':1,'Gd':2,'Ex':3,'Fa':4}
dict_ExterCond={'TA':1,'Gd':2,'Ex':3,'Fa':4,'Po':5}
dict_Foundation={'PConc':1,'CBlock':2,'BrkTil':3,'Slab':4,'Stone':5,'Wood':6}
dict_BsmtQual={'TA':1,'Gd':2,'Ex':3,'Fa':4}
dict_BsmtCond={'TA':1,'Gd':2,'Fa':3,'Po':4}
dict_BsmtExposure={'No':1,'Av':2,'Gd':3,'Mn':4}
dict_BsmtFinType1={'Unf':1,'GLQ':2,'ALQ':3,'BLQ':4,'Rec':5,'LwQ':6}
dict_BsmtFinType2={'Unf':1,'GLQ':2,'ALQ':3,'BLQ':4,'Rec':5,'LwQ':6}
dict_Heating={'GasA':1,'GasW':2,'Grav':3,'Wall':4,'OthW':5,'Floor':6}
dict_HeatingQC={'TA':1,'Gd':2,'Ex':3,'Fa':4,'Po':5}
dict_CentralAir={'Y':1,'N':2}
dict_Electrical={'SBrkr':1,'FuseA':2,'FuseF':3,'FuseP':4,'Mix':5}
dict_KitchenQual={'TA':1,'Gd':2,'Ex':3,'Fa':4}
dict_Functional={'Typ':1,'Min2':2,'Min1':3,'Mod':4,'Maj1':5,'Maj2':6,
                 'Sev':7}
dict_FireplaceQu={'TA':1,'Gd':2,'Ex':3,'Fa':4,'Po':5}
dict_GarageType={'Attchd':1,'Detchd':2,'BuiltIn':3,'Basment':4,'CarPort':5,'2Types':6}
dict_GarageFinish={'Unf':1,'RFn':2,'Fin':3}
dict_GarageQual={'TA':1,'Gd':2,'Ex':3,'Fa':4,'Po':5}
dict_GarageCond={'TA':1,'Gd':2,'Ex':3,'Fa':4,'Po':5}
dict_PavedDrive={'Y':1,'N':2,'P':3}
dict_SaleType={'WD':1,'New':2,'COD':3,'ConLD':4,'ConLw':5,
                 'ConLI':6,'CWD':7,'Oth':8,'Con':9}
dict_SaleCondition={'Normal':1,'Partial':2,'Abnorml':3,'Family':4,'Alloca':5,
                 'AdjLand':6}

hdf['MSZoning_dummy'] = hdf['MSZoning'].replace(dict_MSZoning)
hdf['Street_dummy'] =	hdf['Street'].replace(dict_Street )
hdf['LotShape_dummy'] =	hdf['LotShape'].replace(dict_LotShape )
hdf['LandContour_dummy'] =	hdf['LandContour'].replace(dict_LandContour )
hdf['Utilities_dummy'] =	hdf['Utilities'].replace(dict_Utilities )
hdf['LotConfig_dummy'] =	hdf['LotConfig'].replace(dict_LotConfig )
hdf['LandSlope_dummy'] =	hdf['LandSlope'].replace(dict_LandSlope )
hdf['Neighborhood_dummy'] =	hdf['Neighborhood'].replace(dict_Neighborhood )
hdf['Condition1_dummy'] =	hdf['Condition1'].replace(dict_Condition1)
hdf['Condition2_dummy'] =	hdf['Condition2'].replace(dict_Condition2)
hdf['BldgType_dummy'] =	hdf['BldgType'].replace(dict_BldgType)
hdf['HouseStyle_dummy'] =	hdf['HouseStyle'].replace(dict_HouseStyle)
hdf['RoofStyle_dummy'] =	hdf['RoofStyle'].replace(dict_RoofStyle)
hdf['RoofMatl_dummy'] =	hdf['RoofMatl'].replace(dict_RoofMatl)
hdf['Exterior1st_dummy'] =	hdf['Exterior1st'].replace(dict_Exterior1st )
hdf['Exterior2nd_dummy'] =	hdf['Exterior2nd'].replace(dict_Exterior2nd )
hdf['ExterQual_dummy'] =	hdf['ExterQual'].replace(dict_ExterQual)
hdf['ExterCond_dummy'] =	hdf['ExterCond'].replace(dict_ExterCond)
hdf['Foundation_dummy'] =	hdf['Foundation'].replace(dict_Foundation)
hdf['BsmtQual_dummy'] =	hdf['BsmtQual'].replace(dict_BsmtQual)
hdf['BsmtCond_dummy'] =	hdf['BsmtCond'].replace(dict_BsmtCond)
hdf['BsmtExposure_dummy'] =	hdf['BsmtExposure'].replace(dict_BsmtExposure)
hdf['BsmtFinType1_dummy'] =	hdf['BsmtFinType1'].replace(dict_BsmtFinType1)
hdf['BsmtFinType2_dummy'] =	hdf['BsmtFinType2'].replace(dict_BsmtFinType2)
hdf['Heating_dummy'] =	hdf['Heating'].replace(dict_Heating)
hdf['HeatingQC_dummy'] =	hdf['HeatingQC'].replace(dict_HeatingQC)
hdf['CentralAir_dummy'] =	hdf['CentralAir'].replace(dict_CentralAir)
hdf['Electrical_dummy'] =	hdf['Electrical'].replace(dict_Electrical)
hdf['KitchenQual_dummy'] =	hdf['KitchenQual'].replace(dict_KitchenQual)
hdf['Functional_dummy'] =	hdf['Functional'].replace(dict_Functional)
hdf['FireplaceQu_dummy'] =	hdf['FireplaceQu'].replace(dict_FireplaceQu)
hdf['GarageType_dummy'] =	hdf['GarageType'].replace(dict_GarageType)
hdf['GarageFinish_dummy'] =	hdf['GarageFinish'].replace(dict_GarageFinish)
hdf['GarageQual_dummy'] =	hdf['GarageQual'].replace(dict_GarageQual)
hdf['GarageCond_dummy'] =	hdf['GarageCond'].replace(dict_GarageCond)
hdf['PavedDrive_dummy'] =	hdf['PavedDrive'].replace(dict_PavedDrive)
hdf['SaleType_dummy'] =	hdf['SaleType'].replace(dict_SaleType)
hdf['SaleCondition_dummy'] =	hdf['SaleCondition'].replace(dict_SaleCondition)

### Defining final predictors 
predictors = ['MSSubClass', 'MSZoning_dummy', 'LotFrontage', 'LotArea', 'Street_dummy', 'Alley_dummy',  
       'LotShape_dummy', 'LandContour_dummy', 'Utilities_dummy', 'LotConfig_dummy', 'LandSlope_dummy',
       'Neighborhood_dummy', 'Condition1_dummy', 'Condition2_dummy', 'BldgType_dummy', 'HouseStyle_dummy',
       'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle_dummy',
       'RoofMatl_dummy', 'Exterior1st_dummy', 'Exterior2nd_dummy', 
       'ExterQual_dummy', 'ExterCond_dummy', 'Foundation_dummy', 'BsmtQual_dummy', 'BsmtCond_dummy',
       'BsmtExposure_dummy', 'BsmtFinType1_dummy', 'BsmtFinSF1', 'BsmtFinType2_dummy',
       'BsmtUnfSF', 'TotalBsmtSF', 'Heating_dummy', 'HeatingQC_dummy',
       'CentralAir_dummy', 'Electrical_dummy', '1stFlrSF', '2ndFlrSF',
       'GrLivArea', 'BedroomAbvGr', 'KitchenQual_dummy', 'TotRmsAbvGrd',
       'Functional_dummy', 'Fireplaces', 'FireplaceQu_dummy', 'GarageType_dummy', 'GarageYrBlt',
       'GarageFinish_dummy', 'GarageCars', 'GarageArea', 'GarageQual_dummy', 'GarageCond_dummy',
       'PavedDrive_dummy', 'WoodDeckSF', 'OpenPorchSF', 
       'PoolArea','PoolQC_dummy','MiscFeature_dummy',
       'MoSold', 'YrSold', 'SaleType_dummy', 'SaleCondition_dummy']


X = hdf[predictors]
y = hdf['SalePrice']
y.name
hdf['MSSubClass'].dtype
X = X.apply(lambda z : z.astype(int) if z.name in (X.select_dtypes(include='category')) else z)
t=X.dtypes
X_train , X_test, Y_train, Y_test = train_test_split(X,y,random_state=0)

RFR = RandomForestRegressor()
RFR.fit(X_train, Y_train)
RFR_preds = pd.DataFrame(RFR.predict(X_test),columns=['salePrice'],index=Y_test.index)
print(mean_absolute_error(Y_test, RFR_preds))

XGB = XGBRegressor()
XGB.fit(X_train, Y_train, verbose=False)
XGB_preds = pd.DataFrame(XGB.predict(X_test),columns=['salePrice'],index=Y_test.index)
print(mean_absolute_error(Y_test,XGB_preds))


GBR = GradientBoostingRegressor()
GBR.fit(X_train, Y_train)
GBR_preds = pd.DataFrame(GBR.predict(X_test),columns=['salePrice'],index=Y_test.index)
print(mean_absolute_error(Y_test,GBR_preds))
