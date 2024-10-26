import pandas as pd
data = pd.read_csv('D://1 Intern\Telco-Customer-Churn.csv')
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
for column in['gender', 'Partner','Dependents', 'PhoneService' ,'PaperlessBilling', 'Churn']:
    data[column] = label_encoder.fit_transform(data[column])
    # catogerical values have more than two values
data = pd.get_dummies(data ,columns=['MultipleLines','InternetService' ,'OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','PaymentMethod','TechSupport','Contract'])
data.drop(columns=['customerID'] , inplace=True)
data.drop(columns=['TotalCharges'] , inplace=True)
print(data.head())
import seaborn as sns
import matplotlib.pyplot as plt

# Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# Distribution of Churn
sns.countplot(x='Churn', data=data)
plt.title('Churn Distribution')
plt.show()

# Monthly Charges vs Churn
sns.boxplot(x='Churn', y='MonthlyCharges', data=data)
plt.title('Monthly Charges vs Churn')
plt.show()
from sklearn.model_selection import train_test_split

# Separate features and target variable
X = data.drop(columns='Churn')
y = data['Churn']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Build the model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
from sklearn.tree import export_graphviz
import graphviz

# Export the tree structure to a dot file
dot_data = export_graphviz(model, out_file=None, feature_names=X.columns, class_names=['No', 'Yes'], filled=True)

# Visualize using graphviz
graph = graphviz.Source(dot_data)
graph.render("decision_tree")  # Saves the tree as a .pdf or image file
graph.view()