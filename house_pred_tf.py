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
from statsmodels.formula.api import ols
import statsmodels
import statsmodels.api as sm
%matplotlib inline

#read data from the file
house_df = pd.read_csv('house_price_train.csv',index_col=0)

#get summary of data
summary = house_df.describe(include='all')

#get data types of the fields
typ = house_df.dtypes

# conver below fields into object types as per teh field descriptions
house_df['MSSubClass'] = house_df['MSSubClass'].astype('object')
house_df['OverallQual'] = house_df['OverallQual'].astype('object')
house_df['OverallCond'] = house_df['OverallCond'].astype('object')

# perform summary again
summary = house_df.describe(include='all')


### Fields Alley, PoolQC, Fence and MiscFeature has very less count compare to total count
### which means there are too many missing values. Hence we will first check the impact
### of these fields on the target variable and consider them only if they are contributing.
#### Create dummy variables to store these catagorical values (will be removed after analysis if not required)
house_df['hasAlley'] = house_df['Alley'].apply(lambda x: 0 if pd.isnull(x) else 1).astype('object')
house_df['hasFence'] = house_df['Fence'].apply(lambda x: 0 if pd.isnull(x) else 1).astype('object')
house_df['hasPool'] = house_df['PoolQC'].apply(lambda x: 0 if pd.isnull(x) else 1).astype('object')
house_df['hasMiscFeature'] = house_df['MiscFeature'].apply(lambda x: 0 if pd.isnull(x) else 1).astype('object')

### Check summary again
summary = house_df.describe(include='all')

### Check missing values
missing = house_df.isna().sum()

### GarageYrBlt is having missing values. However these values signify that there
### is no garage for that house. Forst we will check the impact of this field on our op
house_df[['GarageYrBlt','SalePrice']].plot('GarageYrBlt','SalePrice',kind='scatter')
### seems to be increasing as the garage is built recently
### hence we will inpute this field with default as 1900 year

house_df['GarageYrBlt'] = house_df['GarageYrBlt'].fillna(1900)

#### Fill NaN (missing values) using Interpolate method for numeric values
#hdf = house_df.apply(lambda x : x.astype('category').cat.codes.replace(-1, np.nan).interpolate(method='linear', limit_direction='both').astype(int).astype('category').cat.rename_categories(x.astype('category').cat.categories) if x.dtype=='object' else x.interpolate())

hdf = house_df.apply(lambda x : x if x.dtype=='object' else x.interpolate())

### Check missing values
missing = hdf.isna().sum()

#### handle missing values for categorical type data
#Field : FireplaceQu
hdf['FireplaceQu'].value_counts()
hdf['FireplaceQu'] = hdf['FireplaceQu'].fillna('NA')
hdf['FireplaceQu'].value_counts()

### create dummy variable as hasFireplace since as per the data info, if fireplace is not present
### the value of this field is NaN. Which is a valid value (NA)
hdf['hasFireplace'] = hdf['Fireplaces'].apply(lambda x: 0 if x == 0 else 1)

# Garage related fields
hdf['GarageType'] = hdf['GarageType'].fillna('NA')
hdf['GarageFinish'] = hdf['GarageFinish'].fillna('NA')
hdf['GarageQual'] = hdf['GarageQual'].fillna('NA')
hdf['GarageCond'] = hdf['GarageCond'].fillna('NA')

### create dummy variable as hasGarage since as per the data info, if Garage is not present
### the value of this field is NaN. Which is a valid value (NA)
hdf['hasGarage'] = hdf['GarageType'].apply(lambda x: 0 if x == 'NA' else 1)

# Basement Related Fields
hdf['BsmtExposure'] = hdf['BsmtExposure'].fillna('NA')
hdf['BsmtFinType2'] = hdf['BsmtFinType2'].fillna('NA')
hdf['BsmtQual'] = hdf['BsmtQual'].fillna('NA')
hdf['BsmtCond'] = hdf['BsmtCond'].fillna('NA')
hdf['BsmtFinType1'] = hdf['BsmtFinType1'].fillna('NA')

### create dummy variable as hasBasement since as per the data info, if basement is not present
### the value of this field is NaN. Which is a valid value (NA)
hdf['hasBasement'] = hdf['BsmtFinType1'].apply(lambda x: 0 if x == 'NA' else 1)

# Electrical
### since only 1 value is missing, we will interpolate the missing value instead of 
### just assigning the default or mode
hdf['Electrical']= hdf['Electrical'].astype('category').cat.codes.replace(-1, np.nan).interpolate(method='linear', limit_direction='both').astype(int).astype('category').cat.rename_categories(hdf['Electrical'].astype('category').cat.categories)

#MasVnrType
### since only few value is missing, we will interpolate the missing value instead of 
### just assigning the default or mode
hdf['MasVnrType']= hdf['MasVnrType'].astype('category').cat.codes.replace(-1, np.nan).interpolate(method='linear', limit_direction='both').astype(int).astype('category').cat.rename_categories(hdf['MasVnrType'].astype('category').cat.categories)
hdf['MasVnrType'].value_counts()

### create dummy variable as hasMasVnr since as per the data info, if Masonry Veneer wall 
### is not present the value of this field is 'None'.
hdf['hasMasVnr'] = hdf['MasVnrType'].apply(lambda x: 0 if x == 'None' else 1)

# Alley, PoolQC, Fence and MiscFeature
hdf['PoolQC'] = hdf['PoolQC'].fillna('NA')
hdf['MiscFeature'] = hdf['MiscFeature'].fillna('NA')
hdf['Alley'] = hdf['Alley'].fillna('NA')
hdf['Fence'] = hdf['Fence'].fillna('NA')

typ = hdf.dtypes
### Find impact of Categorical variables on SalePrice 
stats.linregress(hdf['SalePrice'],hdf['MSSubClass'].astype('category').cat.codes+1) 
stats.linregress(hdf['SalePrice'],hdf['MSZoning'].astype('category').cat.codes+1) 
stats.linregress(hdf['SalePrice'],hdf['Alley'].astype('category').cat.codes+1)
stats.linregress(hdf['SalePrice'],hdf['Street'].astype('category').cat.codes+1) #0.11
stats.linregress(hdf['SalePrice'],hdf['LotShape'].astype('category').cat.codes+1) 
stats.linregress(hdf['SalePrice'],hdf['LandContour'].astype('category').cat.codes+1)#0.55
stats.linregress(hdf['SalePrice'],hdf['Utilities'].astype('category').cat.codes+1) #0.58
stats.linregress(hdf['SalePrice'],hdf['LotConfig'].astype('category').cat.codes+1) 
stats.linregress(hdf['SalePrice'],hdf['LandSlope'].astype('category').cat.codes+1) #0.050
stats.linregress(hdf['SalePrice'],hdf['Neighborhood'].astype('category').cat.codes+1) 
stats.linregress(hdf['SalePrice'],hdf['Condition1'].astype('category').cat.codes+1) 
stats.linregress(hdf['SalePrice'],hdf['Condition2'].astype('category').cat.codes+1) #0.77
stats.linregress(hdf['SalePrice'],hdf['BldgType'].astype('category').cat.codes+1) 
stats.linregress(hdf['SalePrice'],hdf['HouseStyle'].astype('category').cat.codes+1) 
stats.linregress(hdf['SalePrice'],hdf['OverallQual'].astype('category').cat.codes+1) 
stats.linregress(hdf['SalePrice'],hdf['OverallCond'].astype('category').cat.codes+1) 
stats.linregress(hdf['SalePrice'],hdf['RoofStyle'].astype('category').cat.codes+1) 
stats.linregress(hdf['SalePrice'],hdf['RoofMatl'].astype('category').cat.codes+1) 
stats.linregress(hdf['SalePrice'],hdf['Exterior1st'].astype('category').cat.codes+1) 
stats.linregress(hdf['SalePrice'],hdf['Exterior2nd'].astype('category').cat.codes+1)
stats.linregress(hdf['SalePrice'],hdf['MasVnrType'].astype('category').cat.codes+1) #pvalue = 0.63
stats.linregress(hdf['SalePrice'],hdf['ExterQual'].astype('category').cat.codes+1)
stats.linregress(hdf['SalePrice'],hdf['ExterCond'].astype('category').cat.codes+1)
stats.linregress(hdf['SalePrice'],hdf['Foundation'].astype('category').cat.codes+1)
stats.linregress(hdf['SalePrice'],hdf['BsmtQual'].astype('category').cat.codes+1)
stats.linregress(hdf['SalePrice'],hdf['BsmtCond'].astype('category').cat.codes+1)
stats.linregress(hdf['SalePrice'],hdf['BsmtExposure'].astype('category').cat.codes+1)
stats.linregress(hdf['SalePrice'],hdf['BsmtFinType1'].astype('category').cat.codes+1)
stats.linregress(hdf['SalePrice'],hdf['BsmtFinType2'].astype('category').cat.codes+1)
stats.linregress(hdf['SalePrice'],hdf['Heating'].astype('category').cat.codes+1)
stats.linregress(hdf['SalePrice'],hdf['HeatingQC'].astype('category').cat.codes+1)
stats.linregress(hdf['SalePrice'],hdf['CentralAir'].astype('category').cat.codes+1)
stats.linregress(hdf['SalePrice'],hdf['Electrical'].astype('category').cat.codes+1)
stats.linregress(hdf['SalePrice'],hdf['KitchenQual'].astype('category').cat.codes+1)
stats.linregress(hdf['SalePrice'],hdf['Functional'].astype('category').cat.codes+1)
stats.linregress(hdf['SalePrice'],hdf['FireplaceQu'].astype('category').cat.codes+1) 
stats.linregress(hdf['SalePrice'],hdf['GarageType'].astype('category').cat.codes+1)
stats.linregress(hdf['SalePrice'],hdf['GarageFinish'].astype('category').cat.codes+1)
stats.linregress(hdf['SalePrice'],hdf['GarageQual'].astype('category').cat.codes+1) 
stats.linregress(hdf['SalePrice'],hdf['GarageCond'].astype('category').cat.codes+1)
stats.linregress(hdf['SalePrice'],hdf['PavedDrive'].astype('category').cat.codes+1)
stats.linregress(hdf['SalePrice'],hdf['PoolQC'].astype('category').cat.codes+1)
stats.linregress(hdf['SalePrice'],hdf['Fence'].astype('category').cat.codes+1)
stats.linregress(hdf['SalePrice'],hdf['MiscFeature'].astype('category').cat.codes+1)
stats.linregress(hdf['SalePrice'],hdf['SaleType'].astype('category').cat.codes+1)
stats.linregress(hdf['SalePrice'],hdf['SaleCondition'].astype('category').cat.codes+1)


### checking impact of engineered fields
stats.linregress(hdf['SalePrice'],hdf['hasAlley'].astype('category').cat.codes+1)
stats.linregress(hdf['SalePrice'],hdf['hasFireplace'].astype('category').cat.codes+1)
stats.linregress(hdf['SalePrice'],hdf['hasGarage'].astype('category').cat.codes+1)
stats.linregress(hdf['SalePrice'],hdf['hasBasement'].astype('category').cat.codes+1)
stats.linregress(hdf['SalePrice'],hdf['hasMasVnr'].astype('category').cat.codes+1)
stats.linregress(hdf['SalePrice'],hdf['hasPool'].astype('category').cat.codes+1)
stats.linregress(hdf['SalePrice'],hdf['hasFence'].astype('category').cat.codes+1)
stats.linregress(hdf['SalePrice'],hdf['hasMiscFeature'].astype('category').cat.codes+1)


### Checking distribution of our output variable
hdf['SalePrice'].plot(kind='kde')
hdf['SalePrice'].plot(kind='box')


#### analysis of continuous fields : impact and outlier removals
#Field: LotFrontage
### cehck correlation
hdf[['LotFrontage','SalePrice']].corr().loc['LotFrontage','SalePrice']
### Check scatter plot
hdf[['LotFrontage','SalePrice']].plot('LotFrontage','SalePrice',kind='scatter')
sns.boxplot(hdf['LotFrontage'])
hdf['LotFrontage'].quantile([0,0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.95, 1])
l_replace = hdf['LotFrontage'].quantile(0.05,interpolation='nearest')
h_replace = hdf['LotFrontage'].quantile(0.95,interpolation='nearest')
hdf['LotFrontage'] = hdf['LotFrontage'].apply(lambda x: l_replace if x<l_replace else x)
hdf['LotFrontage'] = hdf['LotFrontage'].apply(lambda x: h_replace if x>h_replace else x)
sns.boxplot(hdf['LotFrontage'])
hdf[['LotFrontage','SalePrice']].plot('LotFrontage','SalePrice',kind='scatter')

### LotFrontage is not contributing


#Field: LotArea
hdf[['LotArea','SalePrice']].corr().loc['LotArea','SalePrice']
hdf[['LotArea','SalePrice']].plot('LotArea','SalePrice',kind='scatter')
sns.boxplot(hdf['LotArea'])
hdf['LotArea'].quantile([0,0.02, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.95, 1])
l_replace = hdf['LotArea'].quantile(0.02,interpolation='nearest')
h_replace = hdf['LotArea'].quantile(0.95,interpolation='nearest')
hdf['LotArea'] = hdf['LotArea'].apply(lambda x: l_replace if x<l_replace else x)
hdf['LotArea'] = hdf['LotArea'].apply(lambda x: h_replace if x>h_replace else x)
sns.boxplot(hdf['LotArea'])
hdf[['LotArea','SalePrice']].plot('LotArea','SalePrice',kind='scatter')


#Field: YearBuilt
hdf[['YearBuilt','SalePrice']].corr().loc['YearBuilt','SalePrice']
sns.boxplot(hdf['YearBuilt'])
hdf[['YearBuilt','SalePrice']].plot('YearBuilt','SalePrice',kind='scatter')
hdf['YearBuilt'].quantile([0,0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.95, 1])
l_replace = hdf['YearBuilt'].quantile(0.01,interpolation='nearest')
#h_replace = hdf['YearBuilt'].quantile(0.95,interpolation='nearest')
hdf['YearBuilt'] = hdf['YearBuilt'].apply(lambda x: l_replace if x<l_replace else x)
#hdf['YearBuilt'] = hdf['YearBuilt'].apply(lambda x: h_replace if x>h_replace else x)
sns.boxplot(hdf['YearBuilt'])
hdf[['YearBuilt','SalePrice']].plot('YearBuilt','SalePrice',kind='scatter')


#Field: YearRemodAdd
hdf[['YearRemodAdd','SalePrice']].corr().loc['YearRemodAdd','SalePrice']
sns.boxplot(hdf['YearRemodAdd'])
hdf[['YearRemodAdd','SalePrice']].plot('YearRemodAdd','SalePrice',kind='scatter')
#hdf['YearRemodAdd'].quantile([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.95, 1])
#l_replace = hdf['YearRemodAdd'].quantile(0.01,interpolation='nearest')
#h_replace = hdf['YearRemodAdd'].quantile(0.95,interpolation='nearest')
#hdf['YearRemodAdd'] = hdf['YearRemodAdd'].apply(lambda x: l_replace if x<l_replace else x)
#hdf['YearRemodAdd'] = hdf['YearRemodAdd'].apply(lambda x: h_replace if x>h_replace else x)
sns.boxplot(hdf['YearRemodAdd'])

#Field: MasVnrArea
hdf[['MasVnrArea','SalePrice']].corr().loc['MasVnrArea','SalePrice']
sns.boxplot(hdf['MasVnrArea'])
hdf[['MasVnrArea','SalePrice']].plot('MasVnrArea','SalePrice',kind='scatter')
hdf['MasVnrArea'].quantile([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.93, 1])
#l_replace = hdf['YearRemodAdd'].quantile(0.01,interpolation='nearest')
h_replace = hdf['MasVnrArea'].quantile(0.93,interpolation='nearest')
#hdf['YearRemodAdd'] = hdf['YearRemodAdd'].apply(lambda x: l_replace if x<l_replace else x)
hdf['MasVnrArea'] = hdf['MasVnrArea'].apply(lambda x: h_replace if x>h_replace else x)
sns.boxplot(hdf['MasVnrArea'])
hdf[['MasVnrArea','SalePrice']].plot('MasVnrArea','SalePrice',kind='scatter')

#Field : BsmtFinSF1
hdf[['BsmtFinSF1','SalePrice']].corr().loc['BsmtFinSF1','SalePrice']
hdf[['BsmtFinSF1','SalePrice']].plot('BsmtFinSF1','SalePrice',kind='scatter')
sns.boxplot(hdf['BsmtFinSF1'])
hdf['BsmtFinSF1'].quantile([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.95, 1])
#l_replace = hdf['YearRemodAdd'].quantile(0.01,interpolation='nearest')
h_replace = hdf['BsmtFinSF1'].quantile(0.95,interpolation='nearest')
#hdf['YearRemodAdd'] = hdf['YearRemodAdd'].apply(lambda x: l_replace if x<l_replace else x)
hdf['BsmtFinSF1'] = hdf['BsmtFinSF1'].apply(lambda x: h_replace if x>h_replace else x)
sns.boxplot(hdf['BsmtFinSF1'])
hdf[['BsmtFinSF1','SalePrice']].plot('BsmtFinSF1','SalePrice',kind='scatter')


#Field : BsmtFinSF2
hdf[['BsmtFinSF2','SalePrice']].corr().loc['BsmtFinSF2','SalePrice']
hdf[['BsmtFinSF2','SalePrice']].plot('BsmtFinSF2','SalePrice',kind='scatter')
sns.boxplot(hdf['BsmtFinSF2'])


#Field: BsmtUnfSF
hdf[['BsmtUnfSF','SalePrice']].corr().loc['BsmtUnfSF','SalePrice']
sns.boxplot(hdf['BsmtUnfSF'])
hdf[['BsmtUnfSF','SalePrice']].plot('BsmtUnfSF','SalePrice',kind='scatter')
hdf['BsmtUnfSF'].quantile([0,0.02, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.97, 1])
#l_replace = hdf['BsmtUnfSF'].quantile(0.02,interpolation='nearest')
h_replace = hdf['BsmtUnfSF'].quantile(0.97,interpolation='nearest')
#hdf['BsmtUnfSF'] = hdf['BsmtUnfSF'].apply(lambda x: l_replace if x<l_replace else x)
hdf['BsmtUnfSF'] = hdf['BsmtUnfSF'].apply(lambda x: h_replace if x>h_replace else x)
sns.boxplot(hdf['BsmtUnfSF'])
hdf[['BsmtUnfSF','SalePrice']].plot('BsmtUnfSF','SalePrice',kind='scatter')

#Field: TotalBsmtSF
hdf[['TotalBsmtSF','SalePrice']].corr().loc['TotalBsmtSF','SalePrice']
sns.boxplot(hdf['TotalBsmtSF'])
hdf[['TotalBsmtSF','SalePrice']].plot('TotalBsmtSF','SalePrice',kind='scatter')
hdf['TotalBsmtSF'].quantile([0,0.03, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.98, 1])
l_replace = hdf['TotalBsmtSF'].quantile(0.03,interpolation='nearest')
h_replace = hdf['TotalBsmtSF'].quantile(0.98,interpolation='nearest')
hdf['TotalBsmtSF'] = hdf['TotalBsmtSF'].apply(lambda x: l_replace if x<l_replace else x)
hdf['TotalBsmtSF'] = hdf['TotalBsmtSF'].apply(lambda x: h_replace if x>h_replace else x)
sns.boxplot(hdf['TotalBsmtSF'])
hdf[['TotalBsmtSF','SalePrice']].plot('TotalBsmtSF','SalePrice',kind='scatter')

#Field: 1stFlrSF
hdf[['1stFlrSF','SalePrice']].corr().loc['1stFlrSF','SalePrice']
hdf[['1stFlrSF','SalePrice']].plot('1stFlrSF','SalePrice',kind='scatter')
sns.boxplot(hdf['1stFlrSF'])
hdf['1stFlrSF'].quantile([0,0.03, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.98, 1])
#l_replace = hdf['1stFlrSF'].quantile(0.03,interpolation='nearest')
h_replace = hdf['1stFlrSF'].quantile(0.98,interpolation='nearest')
#hdf['1stFlrSF'] = hdf['1stFlrSF'].apply(lambda x: l_replace if x<l_replace else x)
hdf['1stFlrSF'] = hdf['1stFlrSF'].apply(lambda x: h_replace if x>h_replace else x)
sns.boxplot(hdf['1stFlrSF'])
hdf[['1stFlrSF','SalePrice']].plot('1stFlrSF','SalePrice',kind='scatter')


#Field: 2ndFlrSF
hdf[['2ndFlrSF','SalePrice']].corr().loc['2ndFlrSF','SalePrice']
sns.boxplot(hdf['2ndFlrSF'])
hdf[['2ndFlrSF','SalePrice']].plot('2ndFlrSF','SalePrice',kind='scatter')

### Create seperate field as total Area which is addition of 1st and 2nd floor

hdf['TotalArea'] = hdf['1stFlrSF']+hdf['2ndFlrSF']
sns.boxplot(hdf['TotalArea'])
hdf['TotalArea'].quantile([0,0.03, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.98, 1])
#l_replace = hdf['1stFlrSF'].quantile(0.03,interpolation='nearest')
h_replace = hdf['TotalArea'].quantile(0.98,interpolation='nearest')
#hdf['1stFlrSF'] = hdf['1stFlrSF'].apply(lambda x: l_replace if x<l_replace else x)
hdf['TotalArea'] = hdf['TotalArea'].apply(lambda x: h_replace if x>h_replace else x)
sns.boxplot(hdf['TotalArea'])
hdf[['TotalArea','SalePrice']].plot('TotalArea','SalePrice',kind='scatter')

#Field: LowQualFinSF
hdf[['LowQualFinSF','SalePrice']].corr().loc['LowQualFinSF','SalePrice']
hdf[['LowQualFinSF','SalePrice']].plot('LowQualFinSF','SalePrice',kind='scatter')
sns.boxplot(hdf['LowQualFinSF'])
hdf['LowQualFinSF'].value_counts()
hdf['LowQualFinSF_dummy'] = hdf['LowQualFinSF'].apply(lambda x: 0 if x == 0 else 1)
stats.linregress(house_df['SalePrice'],hdf['LowQualFinSF_dummy'].astype('category').cat.codes) # 0.06

#Field: GrLivArea
hdf[['GrLivArea','SalePrice']].corr().loc['GrLivArea','SalePrice']
hdf[['GrLivArea','SalePrice']].plot('GrLivArea','SalePrice',kind='scatter')
sns.boxplot(hdf['GrLivArea'])
hdf['GrLivArea'].quantile([0,0.03, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.97, 1])
#l_replace = hdf['GrLivArea'].quantile(0.03,interpolation='nearest')
h_replace = hdf['GrLivArea'].quantile(0.97,interpolation='nearest')
#hdf['GrLivArea'] = hdf['GrLivArea'].apply(lambda x: l_replace if x<l_replace else x)
hdf['GrLivArea'] = hdf['GrLivArea'].apply(lambda x: h_replace if x>h_replace else x)
sns.boxplot(hdf['GrLivArea'])

# Create seperate field as basement bath. Full + half
hdf['BsmtBath'] = hdf['BsmtFullBath']+(hdf['BsmtHalfBath']*0.5)

# create new field as hasBsmtBath if the basement has bathroom
hdf['hasBsmtBath'] = hdf['BsmtBath'].apply(lambda x: 0 if x == 0 else 1)
stats.linregress(hdf['SalePrice'],hdf['hasBsmtBath'].astype('category').cat.codes+1)


#Field: FullBath
hdf[['FullBath','SalePrice']].corr().loc['FullBath','SalePrice']
sns.boxplot(hdf['FullBath'])
hdf['FullBath'].value_counts()
hdf['FullBath'] = hdf['FullBath'].astype('category')

hdf['numberFullBath'] = hdf['FullBath'].apply(lambda x: 2 if x==2 else 1)
stats.linregress(house_df['SalePrice'],hdf['numberFullBath'].astype('category').cat.codes)

#Field: HalfBath
hdf[['HalfBath','SalePrice']].corr().loc['HalfBath','SalePrice']
sns.boxplot(hdf['HalfBath'])
hdf['HalfBath'].value_counts()
hdf['HalfBath'] = hdf['HalfBath'].astype('category')
stats.linregress(hdf['HalfBath'].astype('category').cat.codes,house_df['SalePrice'])

hdf['hasHalfBath'] = hdf['HalfBath'].apply(lambda x: 0 if x==0 else 1)
stats.linregress(house_df['SalePrice'],hdf['hasHalfBath'].astype('category').cat.codes)

#Field: BedroomAbvGr
hdf[['BedroomAbvGr','SalePrice']].corr().loc['BedroomAbvGr','SalePrice']
sns.boxplot(hdf['BedroomAbvGr'])
hdf['BedroomAbvGr'].value_counts()

hdf['numberBedroomAbvGr'] = hdf['BedroomAbvGr'].apply(lambda x: 1 if x>3 else 2 if x>2 else 3 if x>1 else 3)
hdf['numberBedroomAbvGr'].value_counts()
stats.linregress(house_df['SalePrice'],hdf['numberBedroomAbvGr'].astype('category').cat.codes)
sns.boxplot(hdf['numberBedroomAbvGr'])

#Field: KitchenAbvGr
hdf[['KitchenAbvGr','SalePrice']].corr().loc['KitchenAbvGr','SalePrice']
sns.boxplot(hdf['KitchenAbvGr'])
hdf[['KitchenAbvGr','SalePrice']].plot('KitchenAbvGr','SalePrice',kind='scatter')
hdf['KitchenAbvGr'].value_counts()
hdf['KitchenAbvGr'] = hdf['KitchenAbvGr'].astype('category')
#highly skewed


#Field: TotRmsAbvGrd
hdf[['TotRmsAbvGrd','SalePrice']].corr().loc['TotRmsAbvGrd','SalePrice']
sns.boxplot(hdf['TotRmsAbvGrd'])

hdf['numberTotRmsAbvGrd'] = hdf['TotRmsAbvGrd'].apply(lambda x: 8 if x>=8 else 5 if x<=5 else x)
hdf['numberTotRmsAbvGrd'].value_counts()
hdf[['numberTotRmsAbvGrd','SalePrice']].plot('numberTotRmsAbvGrd','SalePrice',kind='scatter')
stats.linregress(hdf['SalePrice'],hdf['numberTotRmsAbvGrd'].astype('category').cat.codes)

#Field: Fireplaces
hdf[['Fireplaces','SalePrice']].corr().loc['Fireplaces','SalePrice']
sns.boxplot(hdf['Fireplaces'])
hdf['Fireplaces'].value_counts()

### We have already defined a field hasFireplace hence we can ignore this field as it will 
### contribute same as the previous field


#Field: GarageYrBlt
hdf[['GarageYrBlt','SalePrice']].corr().loc['GarageYrBlt','SalePrice']
sns.boxplot(hdf['GarageYrBlt'])
hdf['GarageYrBlt'].quantile([0,0.02, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.95, 1])
l_replace = hdf['GarageYrBlt'].quantile(0.02,interpolation='nearest')
#h_replace = hdf['Fireplaces'].quantile(0.95,interpolation='nearest')
hdf['GarageYrBlt'] = hdf['GarageYrBlt'].apply(lambda x: l_replace if x<l_replace else x)
#hdf['Fireplaces'] = hdf['Fireplaces'].apply(lambda x: h_replace if x>h_replace else x)
sns.boxplot(hdf['GarageYrBlt'])
hdf[['GarageYrBlt','SalePrice']].plot('GarageYrBlt','SalePrice',kind='scatter')


#Field: GarageCars
hdf[['GarageCars','SalePrice']].corr().loc['GarageCars','SalePrice']
sns.boxplot(hdf['GarageCars'])
hdf['GarageCars'].value_counts()
hdf['GarageCars'] = hdf['GarageCars'].apply(lambda x: 1 if x<1 else 3 if x>3 else x)
hdf['GarageCars'].value_counts()
hdf[['GarageCars','SalePrice']].plot('GarageCars','SalePrice',kind='scatter')

#Field: GarageArea
hdf[['GarageArea','SalePrice']].corr().loc['GarageArea','SalePrice']
sns.boxplot(hdf['GarageArea'])
hdf[['GarageArea','SalePrice']].plot('GarageArea','SalePrice',kind='scatter')
hdf['GarageArea'].quantile([0,0.02, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.97, 1])
#l_replace = hdf['GarageYrBlt'].quantile(0.02,interpolation='nearest')
h_replace = hdf['GarageArea'].quantile(0.97,interpolation='nearest')
#hdf['GarageYrBlt'] = hdf['GarageYrBlt'].apply(lambda x: l_replace if x<l_replace else x)
hdf['GarageArea'] = hdf['GarageArea'].apply(lambda x: h_replace if x>h_replace else x)
sns.boxplot(hdf['GarageArea'])
hdf[['GarageArea','SalePrice']].plot('GarageArea','SalePrice',kind='scatter')

#Field: WoodDeckSF
hdf[['WoodDeckSF','SalePrice']].corr().loc['WoodDeckSF','SalePrice']
sns.boxplot(hdf['WoodDeckSF'])
hdf[['WoodDeckSF','SalePrice']].plot('WoodDeckSF','SalePrice',kind='scatter')
hdf['WoodDeckSF'].value_counts()

hdf['hasWoodDeck'] = hdf['WoodDeckSF'].apply(lambda x: 0 if x==0 else 1)
stats.linregress(hdf['SalePrice'],hdf['WoodDeckSF'].astype('category').cat.codes)


#Field: OpenPorchSF
hdf[['OpenPorchSF','SalePrice']].corr().loc['OpenPorchSF','SalePrice']
sns.boxplot(hdf['OpenPorchSF'])
hdf[['OpenPorchSF','SalePrice']].plot('OpenPorchSF','SalePrice',kind='scatter')

hdf['hasOpenPorch'] = hdf['OpenPorchSF'].apply(lambda x: 0 if x==0 else 1)
stats.linregress(hdf['SalePrice'],hdf['hasOpenPorch'].astype('category').cat.codes)


#Field: EnclosedPorch
hdf[['EnclosedPorch','SalePrice']].corr().loc['EnclosedPorch','SalePrice']
sns.boxplot(hdf['EnclosedPorch'])
hdf[['EnclosedPorch','SalePrice']].plot('EnclosedPorch','SalePrice',kind='scatter')
hdf['EnclosedPorch'].value_counts()

hdf['hasEnclosedPorch'] = hdf['EnclosedPorch'].apply(lambda x : 0 if x==0 else 1)
stats.linregress(hdf['SalePrice'],hdf['hasEnclosedPorch'].astype('category').cat.codes)

#Field: 3SsnPorch
hdf[['3SsnPorch','SalePrice']].corr().loc['3SsnPorch','SalePrice']
sns.boxplot(hdf['3SsnPorch'])
hdf[['3SsnPorch','SalePrice']].plot('3SsnPorch','SalePrice',kind='scatter')
hdf['3SsnPorch'].value_counts()
#highly skewed


#Field: ScreenPorch
hdf[['ScreenPorch','SalePrice']].corr().loc['ScreenPorch','SalePrice']
sns.boxplot(hdf['ScreenPorch'])
hdf[['ScreenPorch','SalePrice']].plot('ScreenPorch','SalePrice',kind='scatter')
hdf['ScreenPorch'].value_counts()
#highly skewed

### Create a new field as TotalPoarchArea which will be the addition of all
### types of porches
hdf['TotalPoarchArea'] = hdf['OpenPorchSF']+hdf['EnclosedPorch']+hdf['3SsnPorch']+hdf['ScreenPorch']
hdf[['TotalPoarchArea','SalePrice']].plot('TotalPoarchArea','SalePrice',kind='scatter')

hdf['TotalPoarchArea'].value_counts()
## not impacting

### creating a new field as hasPoarch and checking hte impact
hdf['hasPoarch'] = hdf['TotalPoarchArea'].apply(lambda x: 0 if x==0 else 1)
stats.linregress(hdf['SalePrice'],hdf['hasPoarch'].astype('category').cat.codes)


#Field: PoolArea
hdf[['PoolArea','SalePrice']].corr().loc['PoolArea','SalePrice']
sns.boxplot(hdf['PoolArea'])
hdf['PoolArea'].value_counts()
hdf[['PoolArea','SalePrice']].plot('PoolArea','SalePrice',kind='scatter')
### highly skewed

#Field: MiscVal
hdf[['MiscVal','SalePrice']].corr().loc['MiscVal','SalePrice']
sns.boxplot(hdf['MiscVal'])
hdf[['MiscVal','SalePrice']].plot('MiscVal','SalePrice',kind='scatter')
# highly skewed

#Field: MoSold
hdf[['MoSold','SalePrice']].corr().loc['MoSold','SalePrice']
sns.boxplot(hdf['MoSold'])
stats.linregress(hdf['SalePrice'],hdf['MoSold'].astype('category').cat.codes) #0.07
# p value is >0.05

#Field: YrSold
hdf[['YrSold','SalePrice']].corr().loc['YrSold','SalePrice']
sns.boxplot(hdf['YrSold'])
stats.linregress(hdf['SalePrice'],hdf['YrSold'].astype('category').cat.codes) #0.26
# p value is >0.05


#### analysis of the target variable
sns.boxplot(hdf['SalePrice'])

### Many outliers above the max quantile
hdf['SalePrice'].quantile([0,0.02, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.99, 1])
h_replace = hdf['SalePrice'].quantile(0.99,interpolation='nearest')
hdf['SalePrice'] =  hdf['SalePrice'].apply(lambda x: h_replace if x>h_replace else x)
sns.boxplot(hdf['SalePrice'])

typ = hdf.dtypes

### changing the categorical fields to numbers
hdf['MSSubClass'] = hdf['MSSubClass'].astype('int64')
hdf['OverallQual'] = hdf['OverallQual'].astype('int64')
hdf['OverallCond'] = hdf['OverallCond'].astype('int64')
hdf['hasAlley'] = hdf['hasAlley'].astype('int64')
hdf['hasFence'] = hdf['hasFence'].astype('int64')
hdf['hasPool'] = hdf['hasPool'].astype('int64')
hdf['hasMiscFeature'] = hdf['hasMiscFeature'].astype('int64')
hdf['GarageYrBlt'] = hdf['GarageYrBlt'].astype('int64')
hdf['MasVnrType'] = hdf['MasVnrType'].astype('object')
hdf['Electrical'] = hdf['Electrical'].astype('object')

hdf['MSSubClass'].value_counts()


np.sqrt(3708531000)
qual = { 'NA':0, 'Po': 1, 'Fa' : 2, 'TA' : 3, 'Gd': 4, 'Ex' : 5}

hdf['ExterQual'] = hdf['ExterQual'].replace(qual)
hdf['ExterCond'] = hdf['ExterCond'].replace(qual)
hdf['BsmtQual'] = hdf['BsmtQual'].replace(qual)
hdf['BsmtCond'] = hdf['BsmtCond'].replace(qual)
hdf['HeatingQC'] = hdf['HeatingQC'].replace(qual)
hdf['KitchenQual'] = hdf['KitchenQual'].replace(qual)
hdf['FireplaceQu'] = hdf['FireplaceQu'].replace(qual)
hdf['GarageQual'] = hdf['GarageQual'].replace(qual)
hdf['GarageCond'] = hdf['GarageCond'].replace(qual)
hdf['PoolQC'] = hdf['PoolQC'].replace(qual)

for column in hdf:
    if (hdf[column].dtype == 'object'):
        new_col = column+'_category'
        hdf[new_col] = hdf[column].astype('category').cat.codes+1


summary = hdf.describe(include='all')

typ = hdf.dtypes


### Defining final predictors 
predictors = ['MSSubClass','MSZoning',	'Street',	'Alley',	
'LandContour',	'LandSlope',	'Condition1',	
'Condition2',	'HouseStyle',	'OverallQual', 'OverallCond',
'RoofStyle',	'RoofMatl',	'Exterior1st',	'Exterior2nd',	
'MasVnrType',	'ExterQual',	'ExterCond',	
'BsmtQual',	'BsmtCond',	'BsmtExposure',	
'BsmtFinType1',	'BsmtFinType2',	'Heating',	
'HeatingQC',	'CentralAir',	'Electrical',	
'KitchenQual',	'Functional',	'FireplaceQu',	
'GarageType',	'GarageFinish',	'GarageQual',	
'GarageCond',	'PavedDrive',	'PoolQC',	'Fence',	
'MiscFeature',	'SaleType',	'SaleCondition',	
'hasAlley',	'hasFireplace',	'hasGarage',	'hasBasement',	'hasMasVnr',	'hasPool',	
'hasFence',	'hasMiscFeature', 'LotArea', 'YearBuilt', 'BsmtFinSF1', 'TotalBsmtSF',
'1stFlrSF', 'TotalArea', 'hasBsmtBath', 'numberFullBath', 'hasHalfBath', 'numberBedroomAbvGr',
'numberTotRmsAbvGrd', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'hasWoodDeck', 'hasOpenPorch',
'hasEnclosedPorch','hasPoarch']

X = hdf[predictors]
y = hdf['SalePrice']
#all_col = predictors
#all_col.append('SalePrice')
#final = hdf[all_col]
#final.to_csv('pred_data.csv')
#y.name

X = X.apply(lambda z : z.astype(int) if z.name in (X.select_dtypes(include='category')) else z)
X = X.apply(lambda z : z.astype(int) if z.name in (X.select_dtypes(include='O')) else z)

X_train , X_test, Y_train, Y_test = train_test_split(X,y,random_state=0)

## create files for tensorflow code
df_train = pd.merge(X_train,pd.DataFrame(Y_train, index = X_train.index), how = 'inner', left_on = X_train.index,
                    right_on = Y_train.index)
df_train = df_train.set_index('key_0')

typ = df_train.dtypes
df_train.to_csv('df_train_cat.csv', header=None, index=False)

df_valid = pd.merge(X_test,pd.DataFrame(Y_test, index = X_test.index), how = 'inner', 
                    left_on = X_test.index, right_on = Y_test.index)

df_valid = df_valid.set_index('key_0')

typ = df_valid.dtypes
df_valid.to_csv('df_valid_cat.csv', header=None, index=False)

df_train.columns
RFR = RandomForestRegressor()
RFR.fit(X_train, Y_train)

RFR_preds = pd.DataFrame(RFR.predict(X_test),columns=['salePrice'],index=Y_test.index)
print(mean_absolute_error(Y_test, RFR_preds))

RFR_new = RFR_preds.apply(lambda x: np.power(np.e,x).astype('int64'))

XGB = XGBRegressor()
XGB.fit(X_train, Y_train, verbose=False)
XGB_preds = pd.DataFrame(XGB.predict(X_test),columns=['salePrice'],index=Y_test.index).astype(int)
print(mean_absolute_error(Y_test,XGB_preds))

XGB_new = XGB_preds.apply(lambda x: np.power(np.e,x).astype('int64'))

GBR = GradientBoostingRegressor()
GBR.fit(X_train, Y_train)
GBR_preds = pd.DataFrame(GBR.predict(X_test),columns=['salePrice'],index=Y_test.index)
print(mean_absolute_error(Y_test,GBR_preds))

GBR_new = GBR_preds.apply(lambda x: np.power(np.e,x).astype('int64'))

sns.swarmplot(x=GBR_preds['salePrice'],y=Y_test)
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

kfold = KFold(n_splits=10, random_state=7)
results = cross_val_score(XGB, X, y, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

results2 = cross_val_score(RFR, X, y, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results2.mean()*100, results2.std()*100))

results3 = cross_val_score(GBR, X, y, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results3.mean()*100, results3.std()*100))

XGB.score(X_train, Y_train)
RFR.score(X_train, Y_train)
GBR.score(X_train, Y_train)

XGB.score(X_test, Y_test)
RFR.score(X_test, Y_test)
GBR.score(X_test, Y_test)
