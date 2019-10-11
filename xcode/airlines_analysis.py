########################################
############### Analysis ###############
########################################

############### Imports ###############
import pandas as pd
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures


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

## Checking Distribution of Prices
plt.hist(df['prices'])
print(df['prices'].describe())

############### 1.Features ###############
X = df.loc[:,['departure_time', 'arrival_time', 'airline', 'duration', 'number_stops', 'departure_airport', 'arrival_airport', 'date']]
y = df['prices']

## Prices are skewed left so take log to transform into a Gaussian Dist.
log_y = np.log(y)
plt.hist(log_y);
print(log_y.head())

############### 2.Dummy Variables ###############
## Dummy Variables for Airlines, Departure Airport, Arrival Airport, and Date
print(X['airline'].nunique(), 'airlines') # 10 airlines
print(X['departure_airport'].nunique(), 'departure airports')
print(X['arrival_airport'].nunique(), 'arrival airports')
print(X['number_stops'].nunique(), 'number of stops\n')
X = pd.get_dummies(X)
X.info()


############### 3.Split Data ###############
## Splitting Train, Validate, Test Rows
X, X_test, y_log, y_test = train_test_split(X, log_y, test_size=.2, random_state=10)
X_train, X_val, y_train, y_val = train_test_split(X, log_y, test_size=.25, random_state=33)


############### 4.Scaling & Modeling ###############
# Model 1
lm = LinearRegression()

# Feature Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train.values)
X_val_scaled = scaler.transform(X_val.values)
X_test_scaled = scaler.transform(X_test.values)

# Model 2
lm_reg = Ridge(alpha=1)

# Feature Transform
poly = PolynomialFeatures(degree=2) 

X_train_poly = poly.fit_transform(X_train.values)
X_val_poly = poly.transform(X_val.values)
X_test_poly = poly.transform(X_test.values)

# Model 3
lm_poly = LinearRegression()


############### 5.Choose Model ###############

lm.fit(X_train, y_train)
print(f'Linear Regression val R^2: {lm.score(X_val, y_val):.3f}')

lm_reg.fit(X_train_scaled, y_train)
print(f'Ridge Regression val R^2: {lm_reg.score(X_val_scaled, y_val):.3f}')

lm_poly.fit(X_train_poly, y_train)
print(f'Degree 2 polynomial regression val R^2: {lm_poly.score(X_val_poly, y_val):.3f}')

## Choosing Linear Regression Model


############### Test ###############
lm.fit(X,log_y)
print(f'Linear Regression test R^2: {lm.score(X_test, y_test):.3f}')


############### Predict ###############
pred_train = lm.predict(X_train)
pred_test = lm.predict(X_test)


############### Mean Absolute Error ###############
## Create Predicted y and Actual y Series with equal shape
y_pred = pd.DataFrame(np.exp(pred_test))
print(y_pred.shape)

yy_actual = pd.DataFrame(y.sample(613, random_state=100))
print(y_actual.shape)

mean_absolute_error(y_actual, y_pred)
# On average, my model was off by approximately $

## Remember from EDA of Prices
# count    3064
# mean     $390.92
# std      $125.66
# min        $88
# 25%       $312
# 50%       $379
# 75%       $433
# max      $1486


############### Actual vs Predicted ###############
plt.scatter(pred_test, y_test, alpha=.2)
# plt.plot(np.linspace(4,8,1), np.linspace(4,8,1))
plt.title('Actual vs. Predicted Airline Prices')
plt.xlabel('Predicted Airline Prices')
plt.ylabel('Actual Airline Prices');

