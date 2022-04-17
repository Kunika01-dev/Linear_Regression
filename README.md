<h2>Problem Statement- To apply linear regression algorithm </h2>
<h3>The dataset is taken from Kaggle. Check out the below link<h3>
    <h3>https://www.kaggle.com/ashydv/advertising-dataset</h3>

<br>
<h3>✔️Import the necessary libraries</h3>
  ▶️<i>import numpy as np</i>
  <br>
  ▶️<i>import pandas as pd</i>
  <br>
  ▶️<i>import sklearn as sns</i>
  <br>
  ▶️<i>import matplotlib.pyplot as plt</i>
  <br>
  <h3>✔️Read the csv file containing dataset </h3>
<h3>✔️Keep columns with column name Sales and TV and drop rest other columns</h3>
<h3>✔️Here Sales is a dependent variable and TV is an independent variable </h3>
  <h3>✔️For the data analysis:</h3>
 ▶️ <i>Check the relationship/ dependency of the above two variables using scatterplot and pairplot</i>
  <br>
  <h3>✔️Reshape the dataset if required </h3><br>
  ▶️<i>From sklearn.model_selection import train_test_split</i><br>
  ▶️<i>from sklearn.linear_model import LinearRegression</i><br>
  <h3>✔️Define the train and test dataset </h3><br>
  ▶️<i>X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.3)</i><br>
   <h3>✔️Create an estimator /object. Let's say 'lm'</h3><br>
  ▶️<i>lm=LinearRegression()</i>
  <h3>✔️Fit the train data in estimator</h3><br>
  ▶️<i>lm.fit(X_train,y_train)</i>
  <h3>✔️Predict the values of X_test (test data) by passing in estimator</h3><br>
  ▶️<i>y_pred=lm.predict(X_test)</i>
  <h3>✔️Calculate the slope and intercept values</h3><br>
  ▶️<i>a=lm.coef_</i>
  ▶️<i>b=lm.intercept_</i>
 <h3>✔️Calculate the root mean square error:</h3><br>
    ▶️<i>from sklearn.metrics import mean_squared_error</i>
  ▶️<i>rmse=np.sqrt(mean_squared_error(y_pred,y_test))</i>
<h3>✔️Calculate the R2 square: </h3><br>
  ▶️<i>from sklearn.metrics import r2_score</i>
  ▶️ <i>r2_score(y_test,y_pred)</i>
  
  
  
