import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


data = pd.read_csv('E0.csv')
data = data.drop(['Div'], 1)

n_matches = data.shape[0]
n_features = data.shape[1] - 1
n_homewins = len(data[data.FTR == 'H'])

win_rate = (float(n_homewins) / n_matches) * 100

le = preprocessing.LabelEncoder()

data["Date"] = data["Date"].astype('category')
data["HomeTeam"] = data["HomeTeam"].astype('category')
data["AwayTeam"] = data["AwayTeam"].astype('category')
data["HTR"] = data["HTR"].astype('category')
data["Referee"] = data["Referee"].astype('category')

data["FTR"] = data["FTR"].astype('category')


data["Date"] = data["Date"].cat.codes
data["HomeTeam"] = data["HomeTeam"].cat.codes
data["AwayTeam"] = data["AwayTeam"].cat.codes
data["HTR"] = data["HTR"].cat.codes
data["Referee"] = data["Referee"].cat.codes
data["FTR"] = data["FTR"].cat.codes

x_all = data.drop(['FTR', 'HTR'], 1)


y_all = data['FTR']

x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.1)

regressor = LogisticRegression()
regressor.fit(x_train, y_train)
pred = regressor.predict(x_test)

teams = x_test[['HomeTeam', 'AwayTeam']]
teams = teams.replace([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                      ['ARS', 'BO', 'BHA', 'BU', 'CAR', 'CHE', 'CRY', 'EVE', 'FUL', 'HUD', 'LEI',
                       'LIV', 'MC', 'MU', 'NEW', 'SOU', 'TH', 'WAT', 'WHU', 'WOL'])


teams['Prediction'] = pred
teams['Actual Result'] = y_test

teams = teams.replace([0, 1, 2], ['A', 'D', 'H'])

print(teams)
print(accuracy_score(y_test, pred))
