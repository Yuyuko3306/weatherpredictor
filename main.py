
import datetime
import time
import joblib
import matplotlib.pyplot as plt
import requests
from jsonpath import jsonpath
from pycaret.regression import *
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from RuleAugmentedEstimator import RuleAugmentedEstimator


def get_data():
  data = []
  startdate = "-12-01"
  enddate = "-12-31"
  url = "https://api.worldweatheronline.com/premium/v1/past-weather.ashx"

  for year in range(2009, 2021):
    params = {'q': '31.3114,120.6181',
              'key': '8461abd2c6a5416497d151256220704',
              'format': 'json',
              'date': str(year) + startdate,
              'enddate': str(year) + enddate}
    headers = {'User-Agent': 'Mozilla/4.0'}
    result = requests.get(url, params=params, headers=headers, verify=False)
    data_json = result.json()
    for days in jsonpath(data_json, '$..weather'):
      for day in days:
        maxtempC = jsonpath(day, '$..maxtempC')[0]
        mintempC = jsonpath(day, '$..mintempC')[0]
        avgtempC = jsonpath(day, '$..avgtempC')[0]
        sunHour = jsonpath(day, '$..sunHour')[0]
        date = jsonpath(day, '$..date')[0]
        t = time.strptime(date, "%Y-%m-%d")
        year, month, day = t[0:3]
        daily = {"date": date, "maxtempC": maxtempC, "mintempC": mintempC, "avgtempC": avgtempC, "sunHour": sunHour}
        data.append(daily)

  df = pd.DataFrame(data)
  df.to_csv("Result.csv", index=False, mode='a')


def data_preprocess():
  year = []
  month = []
  day = []
  till = []
  data = pd.read_csv('Result.csv')
  data = pd.DataFrame(data)
  data = data.sort_values(by='date', ascending=True)

  for d in data.date:
    t = datetime.datetime.strptime(d, "%Y-%m-%d")
    year.append(t.year)
    month.append(t.month)
    day.append(t.day)
    till.append((t - datetime.datetime(1, 1, 1)).total_seconds() / 31536000)
  Y = data.avgtempC
  X = data.drop('avgtempC', axis=1)
  X = X.drop('mintempC', axis=1)
  X = X.drop('maxtempC', axis=1)
  X = X.drop('date', axis=1)
  X = X.drop('sunHour', axis=1)
  X.insert(loc=0, column='year', value=year)
  X.insert(loc=1, column='month', value=month)
  X.insert(loc=2, column='day', value=day)
  X.insert(loc=0, column='till', value=till)

  # model_select(X,Y)

  plt.plot(till, Y)
  plt.xlabel("x - year")
  plt.ylabel("y - daily average temperature C")
  plt.show()

  train(X, Y)
  X.insert(loc=0, column='avgtempC', value=Y)
  return X


def model_select(X, Y):
  X.insert(loc=0, column='target', value=Y)
  clf1 = setup(data=X, target='target')
  # return best model
  best = compare_models()
  # return top 3 models based on 'Accuracy'
  top3 = compare_models(n_select=3)
  # return best model based on AUC
  best = compare_models(sort='AUC')  # default is 'Accuracy'


def train(X, Y):
  rules = {"month": [
    ("=", "201601", -2.0),
    ("=", "201807", +1.0)
  ],
    "Region": [
      ("=", "Suzhou", 0.0),
      ("=", "Wuxi", -0.5),
      ("=", "Shanghai", 0.5),
    ],

  }

  X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
  etr = GradientBoostingRegressor(n_estimators=200, random_state=0)
  hybrid_model = RuleAugmentedEstimator(etr, rules)
  hybrid_model.fit(X_train, y_train)
  pred = hybrid_model.predict(X_test)
  print("R2 score:", r2_score(y_test, pred))
  print("mean squared error:", mean_squared_error(y_test, pred))
  plt.figure()
  plt.plot(np.arange(len(y_test)), y_test, 'go-', label='true value')
  plt.plot(np.arange(len(pred)), pred, 'ro-', label='predict value')
  plt.legend()
  plt.show()
  joblib.dump(hybrid_model, 'weather_predictor.pkl')


def predict_weather(data):
  clf = joblib.load('weather_predictor.pkl')
  print("-" * 48)
  print("Enter the details of the date you would like to predict")
  print("\n")
  option = input("Year: ")
  year = option
  option = input("Month number (00): ")
  month = option
  option = input("Day number (00): ")
  theday = option

  day = str(year) + "-" + str(month) + "-" + str(theday)
  day = datetime.datetime.strptime(day, "%Y-%m-%d")
  till = (day - datetime.datetime(1, 1, 1)).total_seconds() / 31536000

  X = pd.DataFrame([[till, year, month, theday]], columns=["till", "year", "month", "day"])
  print("\n")
  print("-" * 48)
  print("The temperature is predicted to be: " + str(clf.predict(X)[0]))
  print("The temperature was actually: " + str(get_the_weather(data, till)))
  print("-" * 48)
  print("\n")


def get_the_weather(data, till):
  data = data[data['till'] == till]
  return data.avgtempC.values[0]


if __name__ == '__main__':
  data = data_preprocess()
  while True:
    predict_weather(data)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
