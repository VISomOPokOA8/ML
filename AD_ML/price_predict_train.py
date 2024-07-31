import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
import matplotlib.pyplot as plt

plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.figure(figsize=(12, 10))

# 2x2
for i in range(4):
    data = pd.read_csv(f'datas/historical_price/{i}.csv')
    date = data[['date']]
    price = data['price']
    x_train, x_test, y_train, y_test = train_test_split(date, price, test_size=0.2, random_state=0)

    svr = SVR()
    param_grid = {
        'kernel': ['rbf'],
        'C': [0.1, 1, 10, 50, 100, 200, 500, 1000],
        'epsilon': [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0],
        'gamma': ['scale']
    }
    grid_search = GridSearchCV(svr, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(x_train, y_train)

    print(f"Best parameters found for group {i}: ", grid_search.best_params_)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(x_test)

    with open(f'models/predict/{i}.pickle', 'wb') as f:
        pickle.dump(best_model, f)

    # Mean squared error
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean squared error for group {i}: {mse}')
    print('\n')

    # Sorting data
    df_test = pd.DataFrame({'date': x_test.squeeze(), 'price': y_test})
    df_test = df_test.sort_values(by='date')

    df_predict = pd.DataFrame({'date': x_test.squeeze(), 'price_predict': y_pred})
    df_predict = df_predict.sort_values(by='date')

    # Create a subplot
    plt.subplot(2, 2, i + 1)
    plt.plot(df_test['date'], df_test['price'], label='Actual Price')
    plt.plot(df_predict['date'], df_predict['price_predict'], label='Predicted Price')
    plt.xlabel('Date / days')
    plt.ylabel('Price')
    plt.title(f'SVR Regression for Group {i}')
    plt.legend()

plt.tight_layout()
plt.show()
