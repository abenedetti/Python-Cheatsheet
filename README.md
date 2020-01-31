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
`train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)`

#### *Specify Model*
`tree_model = DecisionTreeRegressor(random_state=1)`

`rf_model = RandomForestRegressor(random_state=1)`

#### *Fit Model with train data*
`a_model.fit(train_X, train_y)`

#### *Make validation predictions and calculate mean absolute error*
`val_predictions = a_model.predict(val_X)`
`val_mae = mean_absolute_error(val_predictions, val_y)`

### 3) Intermediate machine learning toolbox (from [Kaggle mini course](https://www.kaggle.com/learn/overview))

#### *Missing values*
**Get names of columns with missing values**<br>
`cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]`

**Approach 1: drop columns in training and validation data**<br>
`reduced_X_train = X_train.drop(cols_with_missing, axis=1)`<br>
`reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)`<br>

**Approach 2: impute missing values**<br>
`from sklearn.impute import SimpleImputer`<br>
`my_imputer = SimpleImputer()`<br>
`imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))`<br>
`imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))`<br>

<sub>We can choose different strategies beside the `mean`. By `fit` the imputer calculates the means of columns from some data, and by `transform` it applies those means to some data (which is just replacing missing values with the means). If both these data are the same (i.e. the data for calculating the means and the data that means are applied to) you can use `fit_transform` which is basically a `fit` followed by a `transform`. Credits for this note in this [post](https://datascience.stackexchange.com/questions/12321/difference-between-fit-and-fit-transform-in-scikit-learn-models).</sub>

*Imputation removed column names; put them back*<br>
`imputed_X_train.columns = X_train.columns`<br>
`imputed_X_valid.columns = X_valid.columns`<br>

**Approach 3: extended imputation (keeping track of which values were imputed)**<br>
*Make copy to avoid changing original data (when imputing)*<br>
`X_train_plus = X_train.copy()`<br>
`X_valid_plus = X_valid.copy()`<br>

*Make new columns indicating what will be imputed*<br>
`for col in cols_with_missing:`<br>
    `X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()`<br>
    `X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()`<br>

*Imputation*<br>
`my_imputer = SimpleImputer()`<br>
`imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))`<br>
`imputed_X_valid_plus = pd.DataFrame(my_imputer.transform(X_valid_plus))`<br>

*Imputation removed column names; put them back*<br>
`imputed_X_train_plus.columns = X_train_plus.columns`<br>
`imputed_X_valid_plus.columns = X_valid_plus.columns`<br>






