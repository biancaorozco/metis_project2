########################################
############### Analysis ###############
########################################

############### Imports ###############
import pandas as pd
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
import statsmodels.api as sm 

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import make_pipeline
from sklearn_pandas import DataFrameMapper


############### Data ###############
df1 = pd.read_csv("/Users/biancaorozco/Desktop/Metis/week02/project-02/airfare_project2/data/expedia_10_8_2019.csv")
df2 = pd.read_csv("/Users/biancaorozco/Desktop/Metis/week02/project-02/airfare_project2/data/expedia_10_9_2019.csv")
df3 = pd.read_csv("/Users/biancaorozco/Desktop/Metis/week02/project-02/airfare_project2/data/expedia_10_10_2019.csv")

frames = [df1, df2, df3]
na_df = pd.concat(frames)
print(na_df.shape)
na_df.sample(10, random_state=50)


############### Functions ###############
## Duration Time to Minutes
def duration_to_min(duration):
    time = duration.split()
    try:
        minutes = int(time[0])*60 + int(time[1])
        return minutes
    except:
        return None

## 12-hr Time to Minutes
def time_to_min(times):
    ztime = times.zfill(7)
    spaced_time = ztime[:5] + ' ' + ztime[5:]
    split = spaced_time.split()
    try:
        if split[2] == 'pm':
            minutes = (int(split[0])+12)*60 + int(split[1])
        else:
            minutes = int(split[0])*60 + int(split[1])
        return minutes
    except:
        return spaced_time


############### Cleaning Data ###############
## Dropping rows with NaNs
na_df.isna().sum()
df = na_df.dropna()
df.isna().sum()

## Converting Flight ID to Integers
df['id_flight'] = pd.to_numeric(df.id_flight, downcast='signed')

## Replacing Number of Stops
df['number_stops'].replace('(Nonstop)', value=int('0'), regex=True, inplace=True)
df['number_stops'].replace('(1 stop)', value=int('1'), regex=True, inplace=True)
df['number_stops'].replace('(2 stop)', value=int('2'), regex=True, inplace=True)
df['number_stops'].replace('(3 stop)', value=int('3'), regex=True, inplace=True)
            
## Removing $ and , from Prices and Convert to Numeric
df['prices'].replace('\$', value='', regex=True, inplace=True)
df['prices'].replace(',', value='', regex=True, inplace=True)
df['prices'] = pd.to_numeric(df.prices)

## Removing h & m from Duration Times
df['duration'].replace('h', value='', regex=True, inplace=True)
df['duration'].replace('m', value='', regex=True, inplace=True)

## Replacing : with _ in Departure and Arrival Times
df['departure_time'].replace(':', value=' ', regex=True, inplace=True)
df['arrival_time'].replace(':', value=' ', regex=True, inplace=True)

## Executing Cleaning Functions
df['departure_time'] = df['departure_time'].apply(time_to_min)
df['arrival_time'] = df['arrival_time'].apply(time_to_min)
df['duration'] = df['duration'].apply(duration_to_min)


############### EDA ###############
df.info()
df.shape 

## Everything Compared to Everything
sns.pairplot(df, hue='airline', diag_kind='kde');

## Barplot of Prices by Airline
plt.figure(figsize=(18, 7))
sns.barplot(x='airline', y='prices', data=df)
plt.xlabel('Airlines')
plt.ylabel('Prices');

## Barplot of Prices by Number of Stops
plt.figure(figsize=(6, 6))
sns.barplot(x='number_stops', y='prices', data=df)
plt.xlabel('Number of Stops')
plt.ylabel('Prices');

## Checking Flight Counts by City Pairs
gb = df.groupby(['departure_airport', 'arrival_airport'])
aggreg = gb.agg({'number_stops':['count']})
aggreg


############### Features ###############
X = df.loc[:,['departure_time', 'arrival_time', 'airline', 'duration', 'number_stops', 'departure_airport', 'arrival_airport', 'date']]
y = df['prices']


############### Feature Engineering  ###############
## Interaction Effects
X2 = X.copy()
# The number of stops effects the duration of the flight
X2['dur_x_stops'] = X2['duration'] * X2['number_stops']


############### Dummy Variables ###############
## Dummy Variables for Airlines, Departure Airport, Arrival Airport, and Date
print(X['airline'].nunique(), 'airlines') # 10 airlines
print(X['departure_airport'].nunique(), 'departure airports')
print(X['arrival_airport'].nunique(), 'arrival airports\n')
X2 = pd.get_dummies(X)
X2.info()


############### Split Data ###############
# X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=.2, random_state=12)

X, X_test, y, y_test = train_test_split(X, y, test_size=.2, random_state=10)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.25, random_state=33)


############### Train/Validation/Test ###############
## First Model
lm = LinearRegression()

#Feature scaling for train, val, and test to run ridge model on each
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train.values)
X_val_scaled = scaler.transform(X_val.values)
X_test_scaled = scaler.transform(X_test.values)

## Second Model
lm_reg = Ridge(alpha=1)

#Feature transforms for train, val, and test to run poly model on each
poly = PolynomialFeatures(degree=2) 

X_train_poly = poly.fit_transform(X_train.values)
X_val_poly = poly.transform(X_val.values)
X_test_poly = poly.transform(X_test.values)

## Third Model
lm_poly = LinearRegression()


###############  ###############

lm.fit(X_train, y_train)
print(f'Linear Regression val R^2: {lm.score(X_val, y_val):.3f}')

lm_reg.fit(X_train_scaled, y_train)
print(f'Ridge Regression val R^2: {lm_reg.score(X_val_scaled, y_val):.3f}')

lm_poly.fit(X_train_poly, y_train)
print(f'Degree 2 polynomial regression val R^2: {lm_poly.score(X_val_poly, y_val):.3f}')

############### K-Folds ###############
lm = LinearRegression()
kf = KFold(n_splits=5, # folds
            shuffle=True,  
            random_state=33)# arbitrary, but important
kf_scores = cross_val_score(lm, X_train, y_train, # estimator, features, target
                cv = kf, # 5 folds
                scoring='r2') # scoring metric
print(np.mean(kf_scores))






posTransform = make_pipeline(FunctionTransformer(np.log), StandardScaler())

mapper = DataFrameMapper([
        (['arrival_time', 'departure_time'], posTransform),
        (['duration'])
])