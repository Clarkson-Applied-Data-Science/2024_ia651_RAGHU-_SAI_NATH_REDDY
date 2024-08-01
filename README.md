# 2024_ia651_RAGHU-_SAI_NATH_REDDY
# **Mobile Usage impact on andacademic performance, health, and overall well-being**
---
## Objective :
The primary goal is to investigate the relationship between students' mobile phone usage and their academic performance, health, and overall well-being. We aim to identify potential correlations and patterns that can shed light on the impact of mobile phone usage on these factors.
---
## Loading Of Packages

* import pandas as pd
* import seaborn as sns
* import matplotlib.pyplot as plt
* from sklearn.model_selection import train_test_split
* from sklearn.preprocessing import LabelEncoder
* from sklearn.ensemble import RandomForestClassifier
* from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
---
## About the Dataset
# Description 
The dataset comprises information about students, their mobile phone usage, and its potential impact on their academic performance and health. Each row represents a student, and columns contain attributes such as demographics, mobile phone details, usage patterns, academic indicators, and health-related factors.
# Origin
The dataset originate from a survey conducted among students, possibly as part of a research study or educational initiative. 
# Load the dataset
df = pd.read_csv('project.csv')
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Names</th>
      <th>Age</th>
      <th>Gender</th>
      <th>Mobile Phone</th>
      <th>Mobile Operating System</th>
      <th>Mobile phone use for education</th>
      <th>Mobile phone activities</th>
      <th>Helpful for studying</th>
      <th>Educational Apps</th>
      <th>Daily usages</th>
      <th>Performance impact</th>
      <th>Usage distraction</th>
      <th>Attention span</th>
      <th>Useful features</th>
      <th>Health Risks</th>
      <th>Beneficial subject</th>
      <th>Usage symptoms</th>
      <th>Symptom frequency</th>
      <th>Health precautions</th>
      <th>Health rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Ali</td>
      <td>21-25</td>
      <td>Male</td>
      <td>Yes</td>
      <td>Android</td>
      <td>Sometimes</td>
      <td>Social Media</td>
      <td>Yes</td>
      <td>Educational Videos</td>
      <td>4-6 hours</td>
      <td>Agree</td>
      <td>During Exams</td>
      <td>Yes</td>
      <td>Camera</td>
      <td>Yes</td>
      <td>Accounting</td>
      <td>Headache</td>
      <td>Never</td>
      <td>Using Blue light filter</td>
      <td>Excellent</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bilal</td>
      <td>21-25</td>
      <td>Male</td>
      <td>Yes</td>
      <td>Android</td>
      <td>Sometimes</td>
      <td>Social Media</td>
      <td>Yes</td>
      <td>Educational Videos</td>
      <td>4-6 hours</td>
      <td>Neutral</td>
      <td>During Exams</td>
      <td>Yes</td>
      <td>Notes Taking App</td>
      <td>Yes</td>
      <td>Browsing Material</td>
      <td>All of these</td>
      <td>Sometimes</td>
      <td>Taking Break during prolonged use</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Hammad</td>
      <td>21-25</td>
      <td>Male</td>
      <td>Yes</td>
      <td>IOS</td>
      <td>Sometimes</td>
      <td>All of these</td>
      <td>Yes</td>
      <td>Educational Videos</td>
      <td>4-6 hours</td>
      <td>Strongly agree</td>
      <td>Not Distracting</td>
      <td>No</td>
      <td>Camera</td>
      <td>Yes</td>
      <td>Browsing Material</td>
      <td>All of these</td>
      <td>Sometimes</td>
      <td>None of Above</td>
      <td>Excellent</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Abdullah</td>
      <td>21-25</td>
      <td>Male</td>
      <td>Yes</td>
      <td>Android</td>
      <td>Frequently</td>
      <td>All of these</td>
      <td>Yes</td>
      <td>Educational Videos</td>
      <td>2-4 hours</td>
      <td>Strongly agree</td>
      <td>During Class Lectures</td>
      <td>No</td>
      <td>Internet Access</td>
      <td>Only Partially</td>
      <td>Reasarch</td>
      <td>NaN</td>
      <td>Never</td>
      <td>Limiting Screen Time</td>
      <td>Excellent</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Waqar</td>
      <td>21-25</td>
      <td>Male</td>
      <td>Yes</td>
      <td>IOS</td>
      <td>Frequently</td>
      <td>All of these</td>
      <td>Yes</td>
      <td>Educational Videos</td>
      <td>&gt; 6 hours</td>
      <td>Agree</td>
      <td>While Studying</td>
      <td>Yes</td>
      <td>Internet Access</td>
      <td>No</td>
      <td>Browsing Material</td>
      <td>Sleep disturbance</td>
      <td>Sometimes</td>
      <td>None of Above</td>
      <td>Excellent</td>
    </tr>
  </tbody>
</table>
</div>

# About Rows

* Names : Student names
* Age : Student ages (in years)
* Gender : Male/Female
* Mobile phone : Do students own a mobile phone? (Yes/No)
* Mobile Operating System : Type of mobile operating system used (e.g. Android, iOS, Other)
* Mobile phone use for education : Do students use their mobile phone for educational purposes?(Sometime/ Frequently /Rarely )
* Mobile phone activities : List of mobile phone activities used for educational purposes (e.g. online research, educational apps, email, online learning platforms)
* Helpful for studying : Do students find mobile phone use helpful for studying? (Yes/No)
* Educational Apps : List of educational apps used.
* Daily usages : Average daily time spent using mobile phone for educational purposes (in hours)
* Performance impact : How does mobile phone use impact academic performance? (Agree/ Neutral/ Strongly agree)
* Usage distraction : Does mobile phone use distract from studying? (During Exams / Not Distracting / During Class Lectures / While Studying )
* Attention span : Has mobile phone use affected attention span? (Yes / No)
* Useful features : What features of mobile phones are useful for learning? (e.g. Internet Access, Camera, Calculator
, Notes Taking App )
* Health Risks : Are students aware of potential health risks associated with excessive mobile phone use? (Yes / No / Only Partially)
* Beneficial subject : Which subjects benefit most from mobile phone use? (e.g. Accounting, Browsing Material, Research)
* Usage symptoms : Are students experiencing any physical or mental symptoms related to mobile phone use? (e.g. Sleep disturbance, headaches, Anxiety or Stress, All of these)
* Symptom frequency : How often are symptoms experienced? (Sometimes / Never / Rarely / Frequently)
* Health precautions : Are students taking precautions to mitigate potential health risks? (Taking Break during prolonged use / Using Blue light filter / Limiting Screen Time / None of Above)
* Health rating : How would students rate their overall physical and mental health? (Excellent / Good / Fair / Poor)

# Data Cleaning

## Handle missing values
df = df.dropna()


<img width="459" alt="Screenshot 2024-07-25 at 9 11 15 PM" src="https://github.com/user-attachments/assets/942f5da6-ea86-47b9-bf3a-158a311a1c80">



## Variable Exploration

# Convert categorical columns to appropriate data types

The code identifies categorical columns in a pandas DataFrame, converts them to the category data type for efficiency and memory optimization
categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    df[col] = df[col].astype('category')
    
---
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
 ## The image shows a count plot of age groups. The most frequent age group is 21-25, followed by 16-20, 26-30, and 31-35.

![image](https://github.com/user-attachments/assets/01d56105-3601-4d45-9680-4262e74f55e4)
## The bar chart showing the distribution of gender.There are two genders represented: Male and Female.
* Male is the more frequent gender in the dataset.
* Female is less frequent compared to Male.

![image](https://github.com/user-attachments/assets/63f71ed7-569c-4c33-89bd-23ac2eb8661d)
![image](https://github.com/user-attachments/assets/7907da9c-45ea-471d-9c2d-a2e0cb2308cb)
## The chart shows how often people use their mobile phones for education. Most people use it sometimes, while fewer people use it frequently or never.

![image](https://github.com/user-attachments/assets/a32b8736-41a4-4e14-bb5f-e8e462d72159)
## The bar chart depicts the count of mobile operating systems.

* Android: The taller bar represents that there are more devices using the Android operating system in the dataset.
* iOS: The shorter bar indicates a smaller number of devices using the iOS operating system compared to Android

![image](https://github.com/user-attachments/assets/43d7caef-64fd-4bbd-a7a6-4a9db925585e)

## The chart shows the frequency of different mobile phone activities.
"All of these" is the most common activity.
Other activities like messaging, social media, and web browsing are less frequent.
(all of these means :- Messaging,Social Media,web-browsing)

![image](https://github.com/user-attachments/assets/95d2865d-a594-4943-8580-b06de257efc3)
## The bar plot shows responses to whether an activity is helpful for studying:

* "No" Bar: Fewer than 10 respondents.
* "Yes" Bar: Around 85 respondents, indicating most find the activity helpful for studying.


The plot highlights a significant preference for "Yes" over "No".

![image](https://github.com/user-attachments/assets/d45b6536-8799-4cf7-99c9-cd92f846c3fc)

## The bar plot shows the usage distribution of different educational apps:

* Educational Videos: Most popular, with over 50 users.
* Language: Used by about 10 users.
*Productivity Tools: Used by about 10 users.
* Study Planner: Used by about 10 users.


Educational videos are significantly more popular compared to other app categories.


![image](https://github.com/user-attachments/assets/e025b28e-aad0-413c-bede-72ad619403d9)

## The chart shows the distribution of daily mobile phone usage.
* Most users spend 4-6 hours on their phones daily.
* A significant portion uses their phones for 2-4 hours.
* Fewer users spend less than 2 hours or more than 6 hours.
  

![image](https://github.com/user-attachments/assets/c462e085-d441-43d9-971d-c7010fe125d5)

## The chart shows the distribution of responses to "Performance impact".
* "Agree" has the highest count, indicating most people agree with the statement.
* "Disagree" and "Strongly disagree" have lower counts.
* "Neutral" and "Strongly agree" have moderate counts.

![image](https://github.com/user-attachments/assets/48df13c3-b50e-4f8b-af6b-21590750fe42)
## The chart shows how often mobile phone usage is distracting in different situations.

* Mobile phones are most distracting "While Studying".
* They are also quite distracting "During Class Lectures".
* Distractions are lower during "Exams" and when phones are "Not Distracting".


![image](https://github.com/user-attachments/assets/4a93da9a-a55c-46b0-8f29-99ebd1f180d8)

## The chart shows the distribution of attention span.
* More people have a good attention span.
* Fewer people have a poor attention span.

![image](https://github.com/user-attachments/assets/3e393abc-13b8-4f8a-a9d3-4c41f311a25a)
## The chart shows Useful feature of mobile phone
* Internet access is the primary feature used on mobile phones, followed by camera, note-taking apps, and calculators.

![image](https://github.com/user-attachments/assets/e002dc73-7899-4b0e-9c2d-725d75f8d587)
## The chart shows Count Plot of Health Risks
* Yes: Highest number of respondents reported health risks.
* Only Partially: Moderate number of respondents reported partial health risks.
* No: Lowest number of respondents reported no health risks.

![image](https://github.com/user-attachments/assets/90bb00bb-6b65-49df-b97e-743c307ed774)
## The chart shows Count Plot of Beneficial Subject
* Research is the most beneficial subject among respondents.
* Browsing Material is the second most beneficial subject.
* Accounting is the least beneficial subject based on the provided data.

![image](https://github.com/user-attachments/assets/05adf848-0784-4d80-803a-2c63df5a1f42)

## The chart shows Usage Symptoms
* Most common: Headache, Sleep disturbance, All of these.
* Less common: Anxiety or Stress.
* Least common: Headache; Sleep disturbance; Anxiety or Stress; All of these

![image](https://github.com/user-attachments/assets/970033c9-214c-4b18-b206-a5e03405648f)

## The chart shows Symptom Frequency
* Most respondents experience symptoms sometimes.
* A significant portion experiences symptoms rarely.
* Fewer respondents experience symptoms never or frequently.
  
![image](https://github.com/user-attachments/assets/9f37880b-506c-4199-9c40-bab1200f2b09)

## The chart shows Health Precautions
* Limiting Screen Time is the most common health precaution.
* None of Above is the second most common response.
* Taking a break during prolonged use is the third most common response.
* Using Blue light filter is the least common health precaution.
  
![image](https://github.com/user-attachments/assets/7d3f31b8-2280-4e66-beb6-8cc673110b77)

## Count Plot of Health Rating
* Most respondents rated their health as Good.
* A significant number rated their health as Excellent.
* Fewer respondents rated their health as Fair or Poor.
* There are very few respondents in the combined categories like Excellent; Good and Excellent; Good; Fair; Poor


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
![image](https://github.com/user-attachments/assets/80d8f894-6774-4f5c-8170-993454f9110a)

## Violin Plot Analysis (Symptom frequency with respect to usage)

### Interpretations:
* Individuals who experience symptoms "frequently" tend to report a wider range of symptoms compared to those who experience symptoms "rarely" or "never."
* The symptom "All of these" is more prevalent among those who experience symptoms "frequently."
* There's an overlap in symptoms across different frequency categories, indicating some commonalities in reported symptoms.   

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
![image](https://github.com/user-attachments/assets/0328fde6-f001-49bd-9f3f-742524671c7d)
## Image Analysis: Health Precautions vs. Daily Usages.

### General Trends:
* Regardless of the health precaution, "All of these" symptoms tend to be more prevalent.
* Symptoms like headache, sleep disturbance, and anxiety appear to be common across different health precaution groups.
* There seems to be a variation in symptom frequency among different health precautions.
  
### Specific Observations:
* The "Using Blue light filter" group shows a relatively higher frequency of "Headache" and "Sleep disturbance" compared to other groups.
*The "None of Above" group exhibits a higher frequency of "All of these" symptoms.

#### Overall, the charts suggest a correlation between certain health precautions and specific usage symptoms, indicating potential areas for further analysis and intervention.







# Contingency table and Chi-square test
contingency_table = pd.crosstab(df['Mobile phone activities'], df['Performance impact'])
chi2, p, dof, expected = chi2_contingency(contingency_table)

print(f'Chi-square statistic: {chi2}')
print(f'P-value: {p}')
print(f'Degrees of freedom: {dof}')
print(f'Expected frequencies: \n{expected}')

<img width="486" alt="Screenshot 2024-07-26 at 12 39 24 AM" src="https://github.com/user-attachments/assets/a3c70564-bc4e-41b4-bb8e-3dfac16305dd">
## Chi-Square Test Results
### Purpose: To determine if there's a relationship between "Mobile phone activities" and "Performance impact".

### Findings:
* Chi-square statistic: 26.79
* P-value: 0.3141
* Degrees of freedom: 24
  
### Interpretation:
* The p-value is greater than the typical significance level of 0.05.
* This indicates that there is no significant association between "Mobile phone activities" and "Performance impact".
* The observed frequencies in the contingency table are not significantly different from the expected frequencies under the assumption of independence.
* 
### Conclusion:
Based on the Chi-square test, there is no evidence to suggest a relationship between mobile phone activities and performance impact in the dataset.
 ![image](https://github.com/user-attachments/assets/04169225-5e6a-45ec-ac88-e64f2db04515)
## Contingency Table Analysis
* The heatmap presents a contingency table showing the relationship between mobile phone activities and their perceived impact on performance.

### Key Observations:
* Most respondents who engage in "All of these" activities agree that it impacts their performance.
* Messaging seems to have the least impact on performance.
* Social Media activities show a mixed impact, with a higher number of respondents agreeing on a negative impact.
* There are relatively few responses for the combined activities and their impact.

### Overall:
*The table suggests a potential correlation between certain mobile phone activities and their perceived impact on performance, with "All of these" activities showing the strongest association with a negative impact.

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
 

# Classification
## Generate predictions
y_pred = model.predict(X_test)
#### Assuming y_test contains your true labels
report = classification_report(y_test, y_pred, zero_division=1)
#### Assuming y_test contains your true labels
report = classification_report(y_test, y_pred, zero_division=1)
print(report)

              precision    recall  f1-score   support

           0       0.27      0.80      0.40         5
           1       0.00      0.00      0.00         1
           2       1.00      0.10      0.18        10
           3       0.50      0.50      0.50         2
           4       1.00      0.00      0.00         1

    accuracy                           0.32        19
   macro avg       0.55      0.28      0.22        19
weighted avg       0.70      0.32      0.25        19

  
# Overfitting and underfitting assessment
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)
print(f'Training Accuracy: {train_accuracy}')
print(f'Test Accuracy: {test_accuracy}')

### If there's a large gap between training and test accuracy, the model might be overfitting
if train_accuracy > test_accuracy + 0.1:
    print("The model might be overfitting. Consider using techniques like cross-validation, pruning, or regularization.")
elif train_accuracy < test_accuracy:
    print("The model might be underfitting. Consider using a more complex model or adding more features.")

    
<img width="839" alt="Screenshot 2024-07-26 at 6 32 06 PM" src="https://github.com/user-attachments/assets/bebbd59a-aa43-441d-a21d-76d31ccb01cf">
.

## Overfitting Detected
* Training Accuracy of 1.0 indicates the model perfectly predicts all instances in the training data.
* Test Accuracy significantly lower suggests the model is not generalizing well to unseen data.

## Potential Solutions:
* Cross-validation: Evaluate model performance on different subsets of the data to get a more reliable estimate.
* Pruning: Simplify the model by removing unnecessary components.
* Regularization: Introduce penalties to the model to prevent overfitting.

# Sample predictions
sample_predictions = model.predict(X_test[:5])
print(f'Sample Predictions: {sample_predictions}')
print(f'Actual Values: {y_test[:5].values}')


<img width="303" alt="Screenshot 2024-07-26 at 6 30 02 PM" src="https://github.com/user-attachments/assets/d1e60201-1af0-4239-b924-5406f885b980">


## Model Predictions vs. Actual Values
* Model Predictions: [3, 0, 0, 0, 1] are the values predicted by the model for the corresponding instances.
*Actual Values: [41, 2, 23, 2, 57, 0, 97, 0, 0, 0] are the true or correct values for the same instances.



# Decision Tree

data2 = pd.DataFrame(data)

### Drop the target column and prepare X and y
X = data.drop(columns=['Names', 'Performance impact'])
y = data['Performance impact']
### Initialize LabelEncoder for categorical columns
label_encoders = {}
for column in X.columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    label_encoders[column] = le

### Encode target variable
le_y = LabelEncoder()
y = le_y.fit_transform(y)

### Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

### Create and fit the Decision Tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

### Predict probabilities
y_pred_proba = model.predict_proba(X_test)

### Visualize the decision tree
plt.figure(figsize=(20,20))

### Convert class names to strings
class_names = [str(cls) for cls in le_y.classes_]
plot_tree(model, feature_names=X.columns, class_names=class_names, filled=True)

plt.title('Decision Tree')
plt.show()
![ Decision_Tree](https://github.com/user-attachments/assets/49b16c29-68c4-4b89-91a1-c77c9e52fed3)


----
# Conclusion
* The analysis aimed to understand the impact of various factors on performance. Factors such as age, gender, mobile phone usage patterns, health conditions, and study habits were considered.

* Preliminary findings suggest that several factors contribute to performance impact. These include mobile phone activities, daily usage, health conditions, and symptoms. 
