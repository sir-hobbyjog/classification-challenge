# Spam Detector (Classification Challenge)

## Description  
This project involves the creation and comparison of two machine learning models, a Logistic Regression model and a Random Forest Classifier, to detect spam. The project utilizes a dataset provided by EdX, specifically designed for spam detection. The goal is to predict which model performs better on this dataset and to understand the influence of various features on spam detection.  

## Installation
To set up this project, you'll need to have Python installed on your system. Additionally, you'll need the following libraries:  

- Pandas
- Scikit-learn  
You can install these libraries using pip:
`pip install pandas scikit-learn`

## Usage
1. Retrieve the dataset from the provided URL.
2. Split the data into training and testing sets.
3. Scale the features using StandardScaler.
4. Create and fit the Logistic Regression and Random Forest Classifier models.
5. Evaluate the models to determine which performs better for spam detection.

## Code Snippets

Splitting data into training and testing sets:  
```
# Assume X and y are already defined  
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

Scaling Features with StandardScaler:  
```
from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
X_train_scaled = scaler.fit_transform(X_train)  
X_test_scaled = scaler.transform(X_test)
```

## Acknowledgements
Thanks to EdX for providing starter code and dataset.
