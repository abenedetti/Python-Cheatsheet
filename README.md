# Python-Cheatsheet

The companion of R-Cheatsheet. :smile:

### 1) Retrieve key with maximum/minimum value in dictionary
`a_dict = {'a':1000, 'b':3000, 'c': 100}`

`max(a_dict, key=a_dict.get)`

`min(a_dict, key=a_dict.get)`
<br>

### 2) Basic machine learning toolbox (from [Kaggle mini course](https://www.kaggle.com/learn/overview))

#### *Load libraries*
`import pandas as pd`<br>
`from sklearn.model_selection import train_test_split`<br>
`from sklearn.tree import DecisionTreeRegressor`<br>
`from sklearn.metrics import mean_absolute_error`<br>
`from sklearn.ensemble import RandomForestRegressor`<br>

#### *Split into validation and training data*
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

#### *Specify Model*
tree_model = DecisionTreeRegressor(random_state=1)
rf_model = RandomForestRegressor(random_state=1)

#### *Fit Model with train data*
a_model.fit(train_X, train_y)

#### *Make validation predictions and calculate mean absolute error*
val_predictions = a_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
