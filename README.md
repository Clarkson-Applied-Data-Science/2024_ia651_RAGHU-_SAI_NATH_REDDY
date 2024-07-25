# 2024_ia651_RAGHU-_SAI_NATH_REDDY
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve

# Load the dataset
df = pd.read_csv('project.csv')

# Handle missing values
df = df.dropna()

# Convert categorical columns to appropriate data types
categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    df[col] = df[col].astype('category')

# Exploratory Data Analysis (EDA)

# Count plots for categorical columns
for i, col in enumerate(categorical_columns):
    if i == 0:
        continue
    plt.figure(figsize=(8, 6))
    sns.countplot(y=col, data=df)
    plt.title(f'Count plot of {col}')
    plt.show()
![image](https://github.com/user-attachments/assets/61d8c183-8314-48eb-8baa-d40fd6e1af7b)

# Symptom frequency vs. usage symptoms
usage_symptoms = ['Never', 'Sometimes', 'Frequently', 'Rarely']
for synp in usage_symptoms:
    symptoms = df[df['Symptom frequency'] == synp]
    usage = symptoms['Usage symptoms'].value_counts().sort_index()

    plt.figure(figsize=(8, 6))
    usage.plot(kind='barh')
    plt.title(f'Symptom frequency with respect to usage {synp}')
    plt.xlabel(synp)
    plt.ylabel('Daily usages')
    plt.xticks(rotation=0)
    plt.show()

# Health precautions vs. usage symptoms
precautions = ['Using Blue light filter', 'Taking Break during prolonged use', 'None of Above', 'Limiting Screen Time']
num_plots = len(precautions)
cols = 2
rows = (num_plots + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(14, 10))
axes = axes.flatten()

for i, prep in enumerate(precautions):
    ax = axes[i]
    precautions_data = df[df['Health precautions'] == prep]
    usage = precautions_data['Usage symptoms'].value_counts().sort_index()

    usage.plot(kind='bar', ax=ax)
    ax.set_title(f'Health precautions {prep} after using mobile for hours')
    ax.set_xlabel(prep)
    ax.set_ylabel('Daily usages')
    ax.set_xticks(range(len(usage.index)))
    ax.set_xticklabels(usage.index, rotation=0)

# Remove empty subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Contingency table and Chi-square test
contingency_table = pd.crosstab(df['Mobile phone activities'], df['Performance impact'])
chi2, p, dof, expected = chi2_contingency(contingency_table)

print(f'Chi-square statistic: {chi2}')
print(f'P-value: {p}')
print(f'Degrees of freedom: {dof}')
print(f'Expected frequencies: \n{expected}')

# Visualize the contingency table using a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(contingency_table, annot=True, cmap="YlGnBu", fmt='d')
plt.title('Contingency Table of Mobile Phone Activities and Performance Impact')
plt.show()

# Encode categorical variables
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Feature selection and train/test split
X = df.drop('Performance impact', axis=1)
y = df['Performance impact']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model selection: Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{class_report}')
print(f'ROC AUC Score: {roc_auc}')

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC Curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Overfitting and underfitting assessment
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)
print(f'Training Accuracy: {train_accuracy}')
print(f'Test Accuracy: {test_accuracy}')

if train_accuracy > test_accuracy + 0.1:
    print("The model might be overfitting. Consider using techniques like cross-validation, pruning, or regularization.")
elif train_accuracy < test_accuracy:
    print("The model might be underfitting. Consider using a more complex model or adding more features.")

# Sample predictions
sample_predictions = model.predict(X_test[:5])
print(f'Sample Predictions: {sample_predictions}')
print(f'Actual Values: {y_test[:5].values}')

# New or synthesized examples for prediction
new_data = pd.DataFrame({
    'Names': ['John Doe', 'Jane Smith'],
    'Age': [20, 25],
    'Gender': ['Male', 'Female'],
    'Mobile Phone': ['Samsung', 'Apple'],
    'Mobile Operating System': ['Android', 'iOS'],
    'Mobile phone use for education': ['Yes', 'No'],
    'Mobile phone activities': ['Games', 'Social Media'],
    'Helpful for studying': ['Yes', 'No'],
    'Educational Apps': ['Yes', 'No'],
    'Daily usages': ['3-4 hours', '1-2 hours'],
    'Usage distraction': ['Yes', 'No'],
    'Attention span': ['Good', 'Average'],
    'Useful features': ['Apps', 'Messaging'],
    'Health Risks': ['Yes', 'No'],
    'Beneficial subject': ['Math', 'Science'],
    'Usage symptoms': ['Headache', 'Eye strain'],
    'Symptom frequency': ['Frequently', 'Sometimes'],
    'Health precautions': ['Using Blue light filter', 'Limiting Screen Time'],
    'Health rating': ['Good', 'Average']
})

# Encode new data
for col in categorical_columns:
    new_data[col] = label_encoders[col].transform(new_data[col])

new_predictions = model.predict(new_data.drop('Performance impact', axis=1, errors='ignore'))
print(f'New Predictions: {new_predictions}')

# Identifying and mitigating overfitting/underfitting
if train_accuracy > test_accuracy + 0.1:
    print("The model might be overfitting. Consider using techniques like cross-validation, pruning, or regularization.")
elif train_accuracy < test_accuracy:
    print("The model might be underfitting. Consider using a more complex model or adding more features.")

# Advice for deployment
print("For deployment, ensure data preprocessing steps are consistent, monitor model performance over time, and regularly update the model with new data. Be cautious of biases in the training data and ensure the model's predictions are fair and unbiased.")

# Precautions
print("Ensure that the model is used within the context it was trained for. Avoid extrapolation to significantly different populations or conditions. Regularly audit the model's predictions to ensure it continues to perform well in real-world




