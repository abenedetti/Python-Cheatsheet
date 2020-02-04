# Python-Cheatsheet

The companion of R-Cheatsheet. :smile:

### 1) Retrieve key with maximum/minimum value in dictionary
`a_dict = {'a':1000, 'b':3000, 'c': 100}`

`max(a_dict, key=a_dict.get)`

`min(a_dict, key=a_dict.get)`
<br>

### 2) Basic machine learning toolbox (from [Kaggle mini course](https://www.kaggle.com/learn/intro-to-machine-learning))

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

### 3) Dealing with missing values (from [Kaggle mini course](https://www.kaggle.com/alexisbcook/missing-values))

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

### 3) Dealing with categorical variables (from [Kaggle mini course](https://www.kaggle.com/alexisbcook/categorical-variables))

**Approach 1 (Drop Categorical Variables)**<br>
`drop_X_train = X_train.select_dtypes(exclude=['object'])`<br>
`drop_X_valid = X_valid.select_dtypes(exclude=['object'])`<br>

**Approach 2 (Label Encoding)**<br>
`from sklearn.preprocessing import LabelEncoder`<br>

*Make copy to avoid changing original data*<br> 
`label_X_train = X_train.copy()`<br>
`label_X_valid = X_valid.copy()`<br>

*Apply label encoder to each column with categorical data*<br>
`label_encoder = LabelEncoder()`<br>
`for col in object_cols:`<br>
`    label_X_train[col] = label_encoder.fit_transform(X_train[col])`<br>
`    label_X_valid[col] = label_encoder.transform(X_valid[col])`<br>

**Approach 3 (One-Hot Encoding)**<br>
`from sklearn.preprocessing import OneHotEncoder`<br>

*Apply one-hot encoder to each column with categorical data*<br>
`OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)`<br>
`OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))`<br>
`OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols]))`<br>

*One-hot encoding removed index; put it back*<br>
`OH_cols_train.index = X_train.index`<br>
`OH_cols_valid.index = X_valid.index`<br>

*Remove categorical columns (will replace with one-hot encoding)*<br>
`num_X_train = X_train.drop(object_cols, axis=1)`<br>
`num_X_valid = X_valid.drop(object_cols, axis=1)`<br>

*Add one-hot encoded columns to numerical features*<br>
`OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)`<br>
`OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)`<br>

### 4) Dealing with pipelines (from [Kaggle mini course](https://www.kaggle.com/alexisbcook/pipelines))

**Step 1: Define Preprocessing Steps**<br>
`from sklearn.compose import ColumnTransformer`<br>
`from sklearn.pipeline import Pipeline`<br>
`from sklearn.impute import SimpleImputer`<br>
`from sklearn.preprocessing import OneHotEncoder`<br>

*Preprocessing for numerical data*<br>
`numerical_transformer = SimpleImputer(strategy='constant')`<br>

*Preprocessing for categorical data*<br>
`categorical_transformer = Pipeline(steps=[`<br>
`    ('imputer', SimpleImputer(strategy='most_frequent')),`<br>
`    ('onehot', OneHotEncoder(handle_unknown='ignore'))`<br>
`])`<br>

*Bundle preprocessing for numerical and categorical data*<br>
`preprocessor = ColumnTransformer(`<br>
`    transformers=[`<br>
`        ('num', numerical_transformer, numerical_cols),`<br>
`        ('cat', categorical_transformer, categorical_cols)`<br>
`    ])`<br>

**Step 2: Define the Model**<br>
`from sklearn.ensemble import RandomForestRegressor`<br>
`model = RandomForestRegressor(n_estimators=100, random_state=0)`<br>

**Step 3: Create and Evaluate the Pipeline**<br>
`from sklearn.metrics import mean_absolute_error`<br>

*Bundle preprocessing and modeling code in a pipeline*`<br>
`my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),`<br>
`                              ('model', model)`<br>
`                             ])`<br>

*Preprocessing of training data, fit model*<br>
`my_pipeline.fit(X_train, y_train)`<br>

*Preprocessing of validation data, get predictions*<br>
`preds = my_pipeline.predict(X_valid)`<br>

*Evaluate the model*<br>
`score = mean_absolute_error(y_valid, preds)`<br>
`print('MAE:', score)`<br>

### 5) Cross-validation (from [Kaggle mini course](https://www.kaggle.com/alexisbcook/cross-validation))

`#Set pipeline`<br>
`from sklearn.ensemble import RandomForestRegressor`<br>
`from sklearn.pipeline import Pipeline`<br>
`from sklearn.impute import SimpleImputer`<br>

`my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer()),`<br>
`                              ('model', RandomForestRegressor(n_estimators=50,`<br>
`                                                              random_state=0))`<br>
`                             ])`<br>
                             
`#Determine cross validation score`<br>                 
`from sklearn.model_selection import cross_val_score`<br>

`*Multiply by -1 since sklearn calculates *negative* MAE*`<br>
`scores = -1 * cross_val_score(my_pipeline, X, y,`<br>
`                              cv=5,`<br>
`                              scoring='neg_mean_absolute_error')`<br>

`print("MAE scores:\n", scores)`<br>

### 6) Gradient boosting (from [Kaggle mini course](https://www.kaggle.com/alexisbcook/xgboost))

`#Base model`<br>
`from xgboost import XGBRegressor`<br>
`my_model = XGBRegressor()`<br>
`my_model.fit(X_train, y_train)`<br>

`from sklearn.metrics import mean_absolute_error`<br>
`predictions = my_model.predict(X_valid)`<br>

`#Parameters tuning`<br>
`my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)`<br>
`my_model.fit(X_train, y_train,`<br>
`             early_stopping_rounds=5,`<br> 
`             eval_set=[(X_valid, y_valid)],`<br>
`             verbose=False)`<br>


