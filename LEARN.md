# Creating a Logistic Regression Model to Predict Absenteeism


```python
#ignore warnings
import warnings
warnings.filterwarnings('ignore')
```

## Import the Relevant Libraries


```python
import numpy as np
import pandas as pd
```

## Load the Data


```python
data_preprocessed = pd.read_csv('Absenteeism_preprocessed.csv')
```


```python
data_preprocessed.head()
```
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Reason 1</th>
      <th>Reason 2</th>
      <th>Reason 3</th>
      <th>Reason 4</th>
      <th>Month Value</th>
      <th>Day of the Week</th>
      <th>Transportation Expense</th>
      <th>Distance to Work</th>
      <th>Age</th>
      <th>Daily Work Load Average</th>
      <th>Body Mass Index</th>
      <th>Education</th>
      <th>Children</th>
      <th>Pets</th>
      <th>Absenteeism Time in Hours</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>7</td>
      <td>1</td>
      <td>289</td>
      <td>36</td>
      <td>33</td>
      <td>239.55</td>
      <td>30</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>1</td>
      <td>118</td>
      <td>13</td>
      <td>50</td>
      <td>239.55</td>
      <td>31</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>7</td>
      <td>2</td>
      <td>179</td>
      <td>51</td>
      <td>38</td>
      <td>239.55</td>
      <td>31</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>3</td>
      <td>279</td>
      <td>5</td>
      <td>39</td>
      <td>239.55</td>
      <td>24</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>7</td>
      <td>3</td>
      <td>289</td>
      <td>36</td>
      <td>33</td>
      <td>239.55</td>
      <td>30</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>



## Create the Targets

In this model, I choose median of *'Absenteeism Time in Hours'* as the threshold. Anyone who is absent for more than the median hours will be considered as *Excessively Absent (1)* otherwise *Moderately Absent (0)*.

The targets have been calssified in two categories, making it a logistic problem. 


```python
data_preprocessed['Absenteeism Time in Hours'].median()
```




    3.0




```python
targets = np.where(data_preprocessed['Absenteeism Time in Hours'] > data_preprocessed['Absenteeism Time in Hours'].median(), 1, 0)
```


```python
targets
```




    array([1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0,
           1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1,
           0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
           0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1,
           0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0,
           0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0,
           1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1,
           0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1,
           1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
           1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1,
           0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0,
           0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0,
           0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1,
           0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0,
           1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0,
           1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1,
           1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1,
           1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1,
           0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1,
           1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1,
           1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1,
           1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1,
           1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0,
           1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0,
           0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0,
           0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1,
           0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0,
           0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1,
           1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0])




```python
data_preprocessed['Excessive Absenteeism'] = targets
```


```python
data_preprocessed.head()
```
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Reason 1</th>
      <th>Reason 2</th>
      <th>Reason 3</th>
      <th>Reason 4</th>
      <th>Month Value</th>
      <th>Day of the Week</th>
      <th>Transportation Expense</th>
      <th>Distance to Work</th>
      <th>Age</th>
      <th>Daily Work Load Average</th>
      <th>Body Mass Index</th>
      <th>Education</th>
      <th>Children</th>
      <th>Pets</th>
      <th>Absenteeism Time in Hours</th>
      <th>Excessive Absenteeism</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>7</td>
      <td>1</td>
      <td>289</td>
      <td>36</td>
      <td>33</td>
      <td>239.55</td>
      <td>30</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>4.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>1</td>
      <td>118</td>
      <td>13</td>
      <td>50</td>
      <td>239.55</td>
      <td>31</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>7</td>
      <td>2</td>
      <td>179</td>
      <td>51</td>
      <td>38</td>
      <td>239.55</td>
      <td>31</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>3</td>
      <td>279</td>
      <td>5</td>
      <td>39</td>
      <td>239.55</td>
      <td>24</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>4.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>7</td>
      <td>3</td>
      <td>289</td>
      <td>36</td>
      <td>33</td>
      <td>239.55</td>
      <td>30</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>2.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### A Comment on the Targets
We check the preprocessed data is balanced or not, *i.e.*, whether there are as many 1(s) present in the data as there are 0(s). 


```python
targets.sum() / targets.shape[0]
```




    0.4014285714285714



There is a roughly 46:54 percent ratio between Number of Ones and Number of Zeroes, which is good enough to proceed for further processing.


```python
data_with_targets = data_preprocessed.drop(['Absenteeism Time in Hours', 'Day of the Week', 'Daily Work Load Average', 'Distance to Work'], axis=1)
```

We drop the *'Absenteeism Time in Hours'* as it is no more needed in any further preocessing.

Also, we drop the *'Day of the Week'*, *'Daily Work Load Average'* and *'Distance to Work'* columns. At the time of training the model, it was observed that these features do not affect the model much as their coefficient values were too small, approximately ~0.0003.


```python
data_with_targets is data_preprocessed
```




    False




```python
data_with_targets.head()
```
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Reason 1</th>
      <th>Reason 2</th>
      <th>Reason 3</th>
      <th>Reason 4</th>
      <th>Month Value</th>
      <th>Transportation Expense</th>
      <th>Age</th>
      <th>Body Mass Index</th>
      <th>Education</th>
      <th>Children</th>
      <th>Pets</th>
      <th>Excessive Absenteeism</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>7</td>
      <td>289</td>
      <td>33</td>
      <td>30</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>118</td>
      <td>50</td>
      <td>31</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>7</td>
      <td>179</td>
      <td>38</td>
      <td>31</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>279</td>
      <td>39</td>
      <td>24</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>7</td>
      <td>289</td>
      <td>33</td>
      <td>30</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## Selecting the Inputs for the Regression


```python
data_with_targets.shape
```




    (700, 12)




```python
data_with_targets.iloc[:, :-1]
```
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Reason 1</th>
      <th>Reason 2</th>
      <th>Reason 3</th>
      <th>Reason 4</th>
      <th>Month Value</th>
      <th>Transportation Expense</th>
      <th>Age</th>
      <th>Body Mass Index</th>
      <th>Education</th>
      <th>Children</th>
      <th>Pets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>7</td>
      <td>289</td>
      <td>33</td>
      <td>30</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>118</td>
      <td>50</td>
      <td>31</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>7</td>
      <td>179</td>
      <td>38</td>
      <td>31</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>279</td>
      <td>39</td>
      <td>24</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>7</td>
      <td>289</td>
      <td>33</td>
      <td>30</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>695</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>179</td>
      <td>40</td>
      <td>22</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>696</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>225</td>
      <td>28</td>
      <td>24</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>697</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>330</td>
      <td>28</td>
      <td>25</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>698</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>5</td>
      <td>235</td>
      <td>32</td>
      <td>25</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>699</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>5</td>
      <td>291</td>
      <td>40</td>
      <td>25</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>700 rows × 11 columns</p>
</div>




```python
unscaled_inputs = data_with_targets.iloc[:, :-1]
```

## Standardizing the Data


```python
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

class CustomScaler(BaseEstimator, TransformerMixin):
    def __init__(self, columns, copy=True, with_mean=True, with_std=True):
        self.columns = columns
        self.copy = copy
        self.with_mean = with_mean
        self.with_std = with_std
    
    def fit(self, X, y=None):
        self.scaler = StandardScaler(copy=self.copy, with_mean=self.with_mean, with_std=self.with_std)
        self.scaler.fit(X[self.columns], y)
        self.mean_ = np.mean(X[self.columns])
        self.var_ = np.var(X[self.columns])
        return self
    
    def transform(self, X, y=None, copy=None):
        init_col_order = X.columns
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)
        X_not_scaled = X.loc[:, ~X.columns.isin(self.columns)]
        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]
```


```python
unscaled_inputs.columns.values
```




    array(['Reason 1', 'Reason 2', 'Reason 3', 'Reason 4', 'Month Value',
           'Transportation Expense', 'Age', 'Body Mass Index', 'Education',
           'Children', 'Pets'], dtype=object)



We omit *'Reason 1'*, *'Reason 2'*, *'Reason 3'*, *'Reason 4'* and *'Education'* columns for standardizing as these are the dummy categorical variables and standardizing them will lose the information they contain.


```python
columns_to_omit = ['Reason 1', 'Reason 2', 'Reason 3', 'Reason 4', 'Education']

columns_to_scale = [x for x in unscaled_inputs.columns.values if x not in columns_to_omit]
```


```python
absenteeism_scaler = CustomScaler(columns_to_scale)
```


```python
absenteeism_scaler.fit(unscaled_inputs)
```
```python
scaled_inputs = absenteeism_scaler.transform(unscaled_inputs)
```


```python
scaled_inputs
```
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Reason 1</th>
      <th>Reason 2</th>
      <th>Reason 3</th>
      <th>Reason 4</th>
      <th>Month Value</th>
      <th>Transportation Expense</th>
      <th>Age</th>
      <th>Body Mass Index</th>
      <th>Education</th>
      <th>Children</th>
      <th>Pets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.030796</td>
      <td>1.005844</td>
      <td>-0.536062</td>
      <td>0.767431</td>
      <td>0</td>
      <td>0.880469</td>
      <td>0.268487</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.030796</td>
      <td>-1.574681</td>
      <td>2.130803</td>
      <td>1.002633</td>
      <td>0</td>
      <td>-0.019280</td>
      <td>-0.589690</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.030796</td>
      <td>-0.654143</td>
      <td>0.248310</td>
      <td>1.002633</td>
      <td>0</td>
      <td>-0.919030</td>
      <td>-0.589690</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.030796</td>
      <td>0.854936</td>
      <td>0.405184</td>
      <td>-0.643782</td>
      <td>0</td>
      <td>0.880469</td>
      <td>-0.589690</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.030796</td>
      <td>1.005844</td>
      <td>-0.536062</td>
      <td>0.767431</td>
      <td>0</td>
      <td>0.880469</td>
      <td>0.268487</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>695</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.568019</td>
      <td>-0.654143</td>
      <td>0.562059</td>
      <td>-1.114186</td>
      <td>1</td>
      <td>0.880469</td>
      <td>-0.589690</td>
    </tr>
    <tr>
      <th>696</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.568019</td>
      <td>0.040034</td>
      <td>-1.320435</td>
      <td>-0.643782</td>
      <td>0</td>
      <td>-0.019280</td>
      <td>1.126663</td>
    </tr>
    <tr>
      <th>697</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.568019</td>
      <td>1.624567</td>
      <td>-1.320435</td>
      <td>-0.408580</td>
      <td>1</td>
      <td>-0.919030</td>
      <td>-0.589690</td>
    </tr>
    <tr>
      <th>698</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>-0.568019</td>
      <td>0.190942</td>
      <td>-0.692937</td>
      <td>-0.408580</td>
      <td>1</td>
      <td>-0.919030</td>
      <td>-0.589690</td>
    </tr>
    <tr>
      <th>699</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>-0.568019</td>
      <td>1.036026</td>
      <td>0.562059</td>
      <td>-0.408580</td>
      <td>0</td>
      <td>-0.019280</td>
      <td>0.268487</td>
    </tr>
  </tbody>
</table>
<p>700 rows × 11 columns</p>
</div>




```python
scaled_inputs.shape
```




    (700, 11)



## Split the Data into Train & Test and Shuffle

### Import the Relevant Module


```python
from sklearn.model_selection import train_test_split
```

### Split


```python
train_test_split(scaled_inputs, targets)
```




    [     Reason 1  Reason 2  Reason 3  Reason 4  Month Value  \
     486         0         0         0         1     0.929019   
     131         0         0         0         1    -1.765648   
     677         1         0         0         0     0.629611   
     431         0         0         0         1     1.527833   
     512         0         0         0         1     0.929019   
     ..        ...       ...       ...       ...          ...   
     284         0         0         0         1     0.629611   
     44          0         0         0         1     0.629611   
     326         0         0         1         0     1.228426   
     409         0         0         0         1    -1.166834   
     634         1         0         0         0    -1.166834   
     
          Transportation Expense       Age  Body Mass Index  Education  Children  \
     486                1.036026  0.562059        -0.408580          0 -0.019280   
     131               -1.574681  0.091435         0.297027          0 -0.919030   
     677               -1.574681  0.091435         0.297027          0 -0.919030   
     431                0.568211 -0.065439        -0.878984          0  2.679969   
     512               -1.574681  0.091435         0.297027          0 -0.919030   
     ..                      ...       ...              ...        ...       ...   
     284               -1.574681  2.130803         1.002633          0 -0.019280   
     44                -1.016322 -0.379188        -0.408580          0  0.880469   
     326                0.190942  0.091435         0.532229          1 -0.019280   
     409                1.005844 -0.536062         0.767431          0  0.880469   
     634               -0.654143  0.248310         1.002633          0 -0.919030   
     
              Pets  
     486  0.268487  
     131 -0.589690  
     677 -0.589690  
     431 -0.589690  
     512 -0.589690  
     ..        ...  
     284 -0.589690  
     44  -0.589690  
     326  0.268487  
     409  0.268487  
     634 -0.589690  
     
     [525 rows x 11 columns],
          Reason 1  Reason 2  Reason 3  Reason 4  Month Value  \
     262         0         0         0         1     0.929019   
     85          1         0         0         0    -1.466241   
     19          0         0         0         1    -0.568019   
     206         0         0         0         1    -1.466241   
     213         0         0         0         0     1.228426   
     ..        ...       ...       ...       ...          ...   
     470         0         0         0         1     0.030796   
     429         0         0         0         1     0.929019   
     111         0         0         1         0     1.527833   
     681         1         0         0         0     0.929019   
     557         0         0         0         1    -0.268611   
     
          Transportation Expense       Age  Body Mass Index  Education  Children  \
     262               -0.654143 -1.006686        -1.819793          1 -0.919030   
     85                -1.016322 -0.379188        -0.408580          0  0.880469   
     19                 0.387122  1.660180         1.237836          0  0.880469   
     206               -1.016322 -0.379188        -0.408580          0  0.880469   
     213                0.854936  0.405184        -0.643782          0  0.880469   
     ..                      ...       ...              ...        ...       ...   
     470                0.356940  0.718933        -0.878984          0 -0.919030   
     429               -0.654143  0.248310         1.002633          0 -0.919030   
     111                0.356940  0.718933        -0.878984          0 -0.919030   
     681                0.040034  0.718933         0.297027          1  0.880469   
     557                0.040034 -1.320435        -0.643782          0 -0.019280   
     
              Pets  
     262 -0.589690  
     85  -0.589690  
     19   0.268487  
     206 -0.589690  
     213 -0.589690  
     ..        ...  
     470 -0.589690  
     429 -0.589690  
     111 -0.589690  
     681  1.126663  
     557  1.126663  
     
     [175 rows x 11 columns],
     array([1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0,
            0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1,
            0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0,
            0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0,
            1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1,
            0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1,
            0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1,
            1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0,
            1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0,
            1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
            0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0,
            0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1,
            1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0,
            0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0,
            1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1,
            1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0,
            0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1,
            0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0,
            0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0,
            1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1,
            0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0,
            1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1]),
     array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0,
            1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1,
            1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1,
            1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
            1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0,
            1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0,
            0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0,
            1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0])]




```python
x_train, x_test, y_train, y_test = train_test_split(scaled_inputs, targets, train_size = 0.8, random_state = 20)
```


```python
print(x_train.shape, y_train.shape)
```

    (560, 11) (560,)
    


```python
print(x_test.shape, y_test.shape)
```

    (140, 11) (140,)
    

## Logistic Regression with sklearn


```python
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
```

### Train the model


```python
reg = LogisticRegression()
```


```python
reg.fit(x_train, y_train)
```
```python
print(f"The accuracy of our logistic regression model on train data is {(reg.score(x_train, y_train)) * 100 :.2f}%")
```

    The accuracy of our logistic regression model on train data is 72.68%
    

### Manually Checking the Accuracy


```python
model_outputs = reg.predict(x_train)
```


```python
model_outputs
```




    array([0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0,
           0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1,
           0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0,
           0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0,
           0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0,
           0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0,
           0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0,
           1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0,
           0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1,
           1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1,
           1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0,
           0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1,
           0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0,
           0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0,
           0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1,
           0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1,
           0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0,
           1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0,
           0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0,
           0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0,
           0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0,
           0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1,
           0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0,
           0, 1, 0, 0, 1, 0, 0, 0, 0, 0])




```python
y_train
```




    array([0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0,
           1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1,
           1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0,
           0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1,
           1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0,
           0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0,
           0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0,
           0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1,
           1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0,
           1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0,
           0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0,
           1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0,
           0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1,
           1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1,
           0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0,
           1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0,
           0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1,
           0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0,
           0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0,
           0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0,
           1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1,
           0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0,
           0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0,
           0, 0, 0, 1, 1, 0, 1, 0, 1, 0])




```python
model_outputs == y_train
```




    array([ True,  True, False, False,  True,  True,  True,  True,  True,
            True, False,  True, False, False,  True,  True,  True,  True,
           False,  True,  True,  True, False,  True,  True,  True,  True,
           False,  True,  True,  True,  True,  True,  True,  True,  True,
            True, False, False,  True,  True,  True,  True,  True, False,
            True,  True,  True, False,  True, False,  True,  True,  True,
            True, False,  True,  True,  True, False,  True,  True,  True,
            True,  True,  True,  True,  True,  True, False,  True,  True,
           False,  True,  True, False,  True,  True,  True, False,  True,
           False,  True, False,  True,  True, False, False, False, False,
            True,  True, False,  True,  True,  True,  True,  True,  True,
            True, False,  True, False,  True,  True, False,  True,  True,
            True,  True,  True,  True, False,  True,  True,  True,  True,
           False,  True, False,  True,  True, False,  True,  True, False,
            True,  True,  True,  True, False, False,  True,  True,  True,
            True, False,  True,  True,  True,  True,  True, False,  True,
            True, False,  True,  True,  True,  True,  True,  True, False,
            True, False,  True,  True, False,  True, False,  True,  True,
           False, False,  True, False,  True, False,  True, False,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True, False,  True, False,  True,  True,  True,
            True, False,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True, False,  True, False,  True,  True, False,
           False,  True,  True,  True,  True,  True,  True,  True, False,
            True, False,  True, False,  True,  True,  True,  True, False,
            True, False,  True,  True,  True,  True,  True,  True, False,
           False, False,  True, False,  True,  True,  True,  True,  True,
            True,  True,  True, False,  True, False, False,  True,  True,
            True,  True,  True,  True,  True, False, False, False,  True,
           False,  True, False,  True,  True,  True,  True,  True,  True,
           False,  True,  True,  True, False,  True,  True, False, False,
           False,  True,  True, False,  True, False,  True, False,  True,
           False,  True,  True,  True,  True,  True, False,  True,  True,
           False,  True, False,  True,  True,  True, False,  True,  True,
            True,  True,  True, False,  True, False,  True,  True, False,
            True,  True, False,  True,  True, False,  True,  True, False,
            True, False,  True,  True,  True,  True, False,  True, False,
            True,  True,  True, False, False,  True,  True,  True,  True,
            True, False,  True,  True,  True, False,  True,  True,  True,
            True,  True,  True, False,  True,  True,  True,  True,  True,
           False,  True,  True, False,  True, False,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True, False,  True,
            True,  True, False, False,  True, False,  True, False,  True,
            True, False,  True,  True,  True, False,  True, False,  True,
            True,  True,  True,  True,  True, False,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True, False,  True, False,  True, False,  True,
            True,  True,  True, False,  True,  True, False,  True, False,
           False,  True,  True,  True,  True, False, False,  True,  True,
            True,  True, False,  True,  True,  True,  True,  True,  True,
           False,  True,  True,  True,  True, False,  True,  True,  True,
            True, False,  True,  True,  True,  True,  True,  True,  True,
            True,  True, False,  True,  True, False, False,  True,  True,
           False, False,  True,  True,  True,  True,  True, False, False,
            True,  True, False, False,  True,  True,  True, False,  True,
            True,  True,  True,  True, False, False,  True,  True,  True,
            True, False,  True,  True, False,  True,  True,  True,  True,
           False,  True,  True, False,  True,  True, False,  True,  True,
           False, False, False,  True,  True, False,  True,  True,  True,
           False,  True, False,  True,  True,  True,  True,  True,  True,
            True, False,  True, False,  True,  True,  True,  True,  True,
            True,  True, False,  True, False,  True,  True, False,  True,
           False,  True])




```python
np.sum(model_outputs == y_train)
```




    407




```python
model_outputs.shape[0]
```




    560




```python
np.sum(model_outputs == y_train) / model_outputs.shape[0]
```




    0.7267857142857143



### Finding the Intercept and Coefficients


```python
reg.intercept_
```




    array([-1.73732977])




```python
reg.coef_
```




    array([[ 2.24788414,  0.95481002,  2.55470232,  0.88319039,  0.05653064,
             0.52228161, -0.05355861,  0.16780546, -0.13098189,  0.30081469,
            -0.14199405]])




```python
unscaled_inputs.columns.values
```




    array(['Reason 1', 'Reason 2', 'Reason 3', 'Reason 4', 'Month Value',
           'Transportation Expense', 'Age', 'Body Mass Index', 'Education',
           'Children', 'Pets'], dtype=object)




```python
feature_name = unscaled_inputs.columns.values
```


```python
summary_table = pd.DataFrame(columns = ['Feature Name'], data = feature_name)

summary_table['Coefficient'] = np.transpose(reg.coef_)

summary_table
```
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Feature Name</th>
      <th>Coefficient</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Reason 1</td>
      <td>2.247884</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Reason 2</td>
      <td>0.954810</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Reason 3</td>
      <td>2.554702</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Reason 4</td>
      <td>0.883190</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Month Value</td>
      <td>0.056531</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Transportation Expense</td>
      <td>0.522282</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Age</td>
      <td>-0.053559</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Body Mass Index</td>
      <td>0.167805</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Education</td>
      <td>-0.130982</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Children</td>
      <td>0.300815</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Pets</td>
      <td>-0.141994</td>
    </tr>
  </tbody>
</table>



```python
summary_table.index += 1
```


```python
summary_table.loc[0] = ['Intercept', reg.intercept_[0]]
```


```python
summary_table = summary_table.sort_index()
```


```python
summary_table
```




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Feature Name</th>
      <th>Coefficient</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Intercept</td>
      <td>-1.737330</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Reason 1</td>
      <td>2.247884</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Reason 2</td>
      <td>0.954810</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Reason 3</td>
      <td>2.554702</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Reason 4</td>
      <td>0.883190</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Month Value</td>
      <td>0.056531</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Transportation Expense</td>
      <td>0.522282</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Age</td>
      <td>-0.053559</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Body Mass Index</td>
      <td>0.167805</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Education</td>
      <td>-0.130982</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Children</td>
      <td>0.300815</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Pets</td>
      <td>-0.141994</td>
    </tr>
  </tbody>
</table>



### Interpreting the Coefficients


```python
summary_table['Odds Ratio'] = np.exp(summary_table.Coefficient)
```


```python
summary_table
```
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Feature Name</th>
      <th>Coefficient</th>
      <th>Odds Ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Intercept</td>
      <td>-1.737330</td>
      <td>0.175990</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Reason 1</td>
      <td>2.247884</td>
      <td>9.467682</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Reason 2</td>
      <td>0.954810</td>
      <td>2.598177</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Reason 3</td>
      <td>2.554702</td>
      <td>12.867469</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Reason 4</td>
      <td>0.883190</td>
      <td>2.418604</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Month Value</td>
      <td>0.056531</td>
      <td>1.058159</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Transportation Expense</td>
      <td>0.522282</td>
      <td>1.685870</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Age</td>
      <td>-0.053559</td>
      <td>0.947850</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Body Mass Index</td>
      <td>0.167805</td>
      <td>1.182707</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Education</td>
      <td>-0.130982</td>
      <td>0.877234</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Children</td>
      <td>0.300815</td>
      <td>1.350959</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Pets</td>
      <td>-0.141994</td>
      <td>0.867626</td>
    </tr>
  </tbody>
</table>




```python
summary_table.sort_values('Odds Ratio', ascending = False)
```
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Feature Name</th>
      <th>Coefficient</th>
      <th>Odds Ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>Reason 3</td>
      <td>2.554702</td>
      <td>12.867469</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Reason 1</td>
      <td>2.247884</td>
      <td>9.467682</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Reason 2</td>
      <td>0.954810</td>
      <td>2.598177</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Reason 4</td>
      <td>0.883190</td>
      <td>2.418604</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Transportation Expense</td>
      <td>0.522282</td>
      <td>1.685870</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Children</td>
      <td>0.300815</td>
      <td>1.350959</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Body Mass Index</td>
      <td>0.167805</td>
      <td>1.182707</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Month Value</td>
      <td>0.056531</td>
      <td>1.058159</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Age</td>
      <td>-0.053559</td>
      <td>0.947850</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Education</td>
      <td>-0.130982</td>
      <td>0.877234</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Pets</td>
      <td>-0.141994</td>
      <td>0.867626</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Intercept</td>
      <td>-1.737330</td>
      <td>0.175990</td>
    </tr>
  </tbody>
</table>



### Testing the Model


```python
print(f"The accuracy of our model on test data is {(reg.score(x_test, y_test)) * 100 :.2f}%")
```

    The accuracy of our model on test data is 69.29%
    


```python
predicted_proba = reg.predict_proba(x_test)
```


```python
predicted_proba
```




    array([[0.76513567, 0.23486433],
           [0.60349236, 0.39650764],
           [0.55424118, 0.44575882],
           [0.78575503, 0.21424497],
           [0.17436896, 0.82563104],
           [0.48371357, 0.51628643],
           [0.42201855, 0.57798145],
           [0.21147517, 0.78852483],
           [0.73780254, 0.26219746],
           [0.77413734, 0.22586266],
           [0.51352938, 0.48647062],
           [0.27879528, 0.72120472],
           [0.14051543, 0.85948457],
           [0.71317762, 0.28682238],
           [0.33442354, 0.66557646],
           [0.58065566, 0.41934434],
           [0.50930019, 0.49069981],
           [0.53042076, 0.46957924],
           [0.40170817, 0.59829183],
           [0.14023971, 0.85976029],
           [0.72449629, 0.27550371],
           [0.78575503, 0.21424497],
           [0.61251857, 0.38748143],
           [0.59633291, 0.40366709],
           [0.35751721, 0.64248279],
           [0.74106361, 0.25893639],
           [0.54661959, 0.45338041],
           [0.85224029, 0.14775971],
           [0.3918149 , 0.6081851 ],
           [0.78575503, 0.21424497],
           [0.61957409, 0.38042591],
           [0.43477072, 0.56522928],
           [0.45421645, 0.54578355],
           [0.50506966, 0.49493034],
           [0.78575503, 0.21424497],
           [0.46461213, 0.53538787],
           [0.73451511, 0.26548489],
           [0.36475303, 0.63524697],
           [0.56313655, 0.43686345],
           [0.45257283, 0.54742717],
           [0.78859059, 0.21140941],
           [0.61189628, 0.38810372],
           [0.74750611, 0.25249389],
           [0.66661388, 0.33338612],
           [0.30423568, 0.69576432],
           [0.53770159, 0.46229841],
           [0.44667265, 0.55332735],
           [0.78000124, 0.21999876],
           [0.72986282, 0.27013718],
           [0.79139858, 0.20860142],
           [0.57929613, 0.42070387],
           [0.75785459, 0.24214541],
           [0.48371357, 0.51628643],
           [0.74622046, 0.25377954],
           [0.27996088, 0.72003912],
           [0.59536438, 0.40463562],
           [0.15866988, 0.84133012],
           [0.76269205, 0.23730795],
           [0.75673536, 0.24326464],
           [0.75360604, 0.24639396],
           [0.44447672, 0.55552328],
           [0.47104597, 0.52895403],
           [0.75337954, 0.24662046],
           [0.33832109, 0.66167891],
           [0.72760368, 0.27239632],
           [0.76816361, 0.23183639],
           [0.92534076, 0.07465924],
           [0.73120148, 0.26879852],
           [0.34440298, 0.65559702],
           [0.69913129, 0.30086871],
           [0.75384156, 0.24615844],
           [0.70655313, 0.29344687],
           [0.23797794, 0.76202206],
           [0.57144591, 0.42855409],
           [0.48148703, 0.51851297],
           [0.78575503, 0.21424497],
           [0.38778905, 0.61221095],
           [0.36854709, 0.63145291],
           [0.40427284, 0.59572716],
           [0.38555027, 0.61444973],
           [0.74429818, 0.25570182],
           [0.91878548, 0.08121452],
           [0.73402962, 0.26597038],
           [0.40183665, 0.59816335],
           [0.54693139, 0.45306861],
           [0.8585208 , 0.1414792 ],
           [0.4668309 , 0.5331691 ],
           [0.59633291, 0.40366709],
           [0.78352074, 0.21647926],
           [0.43477072, 0.56522928],
           [0.78213208, 0.21786792],
           [0.85677441, 0.14322559],
           [0.75899797, 0.24100203],
           [0.76208043, 0.23791957],
           [0.77413734, 0.22586266],
           [0.2478766 , 0.7521234 ],
           [0.75899797, 0.24100203],
           [0.41349656, 0.58650344],
           [0.74300181, 0.25699819],
           [0.76433241, 0.23566759],
           [0.47304188, 0.52695812],
           [0.43477072, 0.56522928],
           [0.42480856, 0.57519144],
           [0.42369057, 0.57630943],
           [0.56729596, 0.43270404],
           [0.59611587, 0.40388413],
           [0.76513567, 0.23486433],
           [0.2478766 , 0.7521234 ],
           [0.36370168, 0.63629832],
           [0.85883888, 0.14116112],
           [0.92127589, 0.07872411],
           [0.20244594, 0.79755406],
           [0.48640329, 0.51359671],
           [0.63146759, 0.36853241],
           [0.47726283, 0.52273717],
           [0.46939383, 0.53060617],
           [0.36084026, 0.63915974],
           [0.27045558, 0.72954442],
           [0.5569645 , 0.4430355 ],
           [0.72786182, 0.27213818],
           [0.77116418, 0.22883582],
           [0.84350983, 0.15649017],
           [0.28221121, 0.71778879],
           [0.6238841 , 0.3761159 ],
           [0.77413734, 0.22586266],
           [0.70620348, 0.29379652],
           [0.78859059, 0.21140941],
           [0.85257006, 0.14742994],
           [0.46262056, 0.53737944],
           [0.72449629, 0.27550371],
           [0.44838294, 0.55161706],
           [0.78859059, 0.21140941],
           [0.78000124, 0.21999876],
           [0.68103582, 0.31896418],
           [0.7502213 , 0.2497787 ],
           [0.51109671, 0.48890329],
           [0.50506966, 0.49493034],
           [0.69555908, 0.30444092],
           [0.74300181, 0.25699819],
           [0.55896824, 0.44103176]])




```python
predicted_proba.shape
```




    (140, 2)




```python
predicted_proba[:, 1]
```




    array([0.23486433, 0.39650764, 0.44575882, 0.21424497, 0.82563104,
           0.51628643, 0.57798145, 0.78852483, 0.26219746, 0.22586266,
           0.48647062, 0.72120472, 0.85948457, 0.28682238, 0.66557646,
           0.41934434, 0.49069981, 0.46957924, 0.59829183, 0.85976029,
           0.27550371, 0.21424497, 0.38748143, 0.40366709, 0.64248279,
           0.25893639, 0.45338041, 0.14775971, 0.6081851 , 0.21424497,
           0.38042591, 0.56522928, 0.54578355, 0.49493034, 0.21424497,
           0.53538787, 0.26548489, 0.63524697, 0.43686345, 0.54742717,
           0.21140941, 0.38810372, 0.25249389, 0.33338612, 0.69576432,
           0.46229841, 0.55332735, 0.21999876, 0.27013718, 0.20860142,
           0.42070387, 0.24214541, 0.51628643, 0.25377954, 0.72003912,
           0.40463562, 0.84133012, 0.23730795, 0.24326464, 0.24639396,
           0.55552328, 0.52895403, 0.24662046, 0.66167891, 0.27239632,
           0.23183639, 0.07465924, 0.26879852, 0.65559702, 0.30086871,
           0.24615844, 0.29344687, 0.76202206, 0.42855409, 0.51851297,
           0.21424497, 0.61221095, 0.63145291, 0.59572716, 0.61444973,
           0.25570182, 0.08121452, 0.26597038, 0.59816335, 0.45306861,
           0.1414792 , 0.5331691 , 0.40366709, 0.21647926, 0.56522928,
           0.21786792, 0.14322559, 0.24100203, 0.23791957, 0.22586266,
           0.7521234 , 0.24100203, 0.58650344, 0.25699819, 0.23566759,
           0.52695812, 0.56522928, 0.57519144, 0.57630943, 0.43270404,
           0.40388413, 0.23486433, 0.7521234 , 0.63629832, 0.14116112,
           0.07872411, 0.79755406, 0.51359671, 0.36853241, 0.52273717,
           0.53060617, 0.63915974, 0.72954442, 0.4430355 , 0.27213818,
           0.22883582, 0.15649017, 0.71778879, 0.3761159 , 0.22586266,
           0.29379652, 0.21140941, 0.14742994, 0.53737944, 0.27550371,
           0.55161706, 0.21140941, 0.21999876, 0.31896418, 0.2497787 ,
           0.48890329, 0.49493034, 0.30444092, 0.25699819, 0.44103176])



## Saving the Model


```python
import pickle
```


```python
with open('absenteeism_model', 'wb') as file:
    pickle.dump(reg, file)
```


```python
with open('absenteeism_scaler', 'wb') as file:
    pickle.dump(absenteeism_scaler, file)
```
