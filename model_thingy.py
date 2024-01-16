import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# df = pd.read_csv("https://raw.githubusercontent.com/wildcard-mask/web-classification/main/Webpages_Classification_test_data_top_1k.csv", low_memory=False)
df = pd.read_csv("https://raw.githubusercontent.com/wildcard-mask/web-classification/main/Webpages_Classification_train_data_50_percent_original.csv", low_memory=False)
model = DecisionTreeClassifier()

#Replace good/bad & yes/no with respective numerical indicators

df['label'] = df['label'].replace('good', 1)
df['label'] = df['label'].replace('bad', 0)
df['https'] = df['https'].replace('yes', 1)
df['https'] = df['https'].replace('no', 0)

# Convert Dtypes
df = df.astype({'https': 'int16', "label": 'int16', "url_len": 'int16'})


# train_data = train_set.drop(columns=['label'])
# train_answers = train_set.drop(columns=["url", "url_len", "tld", "https"])

# test_data = test_set.drop(columns=['label'])
# test_answers = test_set.drop(columns=["url", "url_len", "tld", "https"])

X = df.drop(columns=["label"]) # input data subset
y = df["label"] # output data subset (target)
type(X)

#Encode the Object Columns

features = ["url", "url_len", "tld", "https"]
one_hot = OneHotEncoder(sparse_output=False)
transformer = ColumnTransformer([("one_hot", one_hot, features)], remainder="passthrough")
transformed_X = transformer.fit_transform(X)


# transform_Train_Data = transformer.fit_transform(train_data)
# transform_Test_Data = transformer.fit_transform(test_data)

# transformer = ColumnTransformer([("one_hot", one_hot, ["label"])], remainder="passthrough")
# transform_Train_Answers = transformer.fit_transform(train_answers)
# transform_Test_Answers = transformer.fit_transform(test_answers)


#Set training & testing data and answers

X_train, X_test, y_train, y_test = train_test_split(transformed_X, y, test_size=0.2)

model.fit(X_train,y_train) # Train the model

predictions = model.predict(X_test)
score = accuracy_score(y_test,predictions)
score