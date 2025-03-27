import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

pd.set_option('display.max_columns', 30)

## Data preprocessing ##

# Heart Disease Dataframe
hd_df = pd.read_csv('../data/processed.cleveland.data', header=None)

hd_columns = ['age', 'sex', 'cp', 'trestbps', 
              'chol', 'fbs', 'restecg', 'thalach',
              'exang', 'oldpeak', 'slope', 'ca',
              'thal', 'num']

hd_df.columns = hd_columns
hd_df['num'] = hd_df.loc[hd_df['num'].isin([0, 1]), 'num']
hd_df.replace('?', np.nan, inplace=True)
hd_df.dropna(inplace=True)
hd_df = hd_df.astype(float)

ct = ColumnTransformer([('scaler', StandardScaler(), [0, 3, 9, 4, 7]),
                        ('encoder', OrdinalEncoder(), [2, 6, 7, 10, 11, 12])], remainder='passthrough')
transformed_hd_df = pd.DataFrame(ct.fit_transform(hd_df), columns=ct.get_feature_names_out())
X, y = transformed_hd_df.drop(columns=['remainder__num']), transformed_hd_df['remainder__num']

sampler = SMOTE()
X_sampled, y_sampled = sampler.fit_resample(X, y)

## Model Training ##

X_train, X_test, y_train, y_test = train_test_split(X_sampled, y_sampled, test_size=0.20, random_state=42)

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

## Model Testing ##

clf_report = classification_report(y_test, y_pred)
print('Classification report: \n')
print(clf_report)

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()
plt.show()