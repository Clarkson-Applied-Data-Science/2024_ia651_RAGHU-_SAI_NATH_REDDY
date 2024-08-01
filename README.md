# 2024_ia651_RAGHU-_SAI_NATH_REDDY
# **Mobile Usage impact on andacademic performance, health, and overall well-being**
---
## Objective :
The primary goal is to investigate the relationship between students' mobile phone usage and their academic performance, health, and overall well-being. We aim to identify potential correlations and patterns that can shed light on the impact of mobile phone usage on these factors.
---
## 1. Data Preparation

---
### About the Dataset
#### Description 
The dataset comprises information about students, their mobile phone usage, and its potential impact on their academic performance and health. Each row represents a student, and columns contain attributes such as demographics, mobile phone details, usage patterns, academic indicators, and health-related factors.
#### Origin
The dataset originate from a survey conducted among students, possibly as part of a research study or educational initiative. 
#### About Rows

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

### Data Cleaning

#### Handle missing values
<img width="459" alt="Screenshot 2024-07-25 at 9 11 15 PM" src="https://github.com/user-attachments/assets/942f5da6-ea86-47b9-bf3a-158a311a1c80">

### Variable Exploration

#### Convert categorical columns to appropriate data types

The code identifies categorical columns in a pandas DataFrame, converts them to the category data type for efficiency and memory optimization
categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    df[col] = df[col].astype('category')
    
---------
## 2.Exploratory Data Analysis (EDA)
--------
### Count plots for categorical columns
for i, col in enumerate(categorical_columns):
    if i == 0:
        continue
    plt.figure(figsize=(8, 6))
    sns.countplot(y=col, data=df)
    plt.title(f'Count plot of {col}')
    plt.show()
    
   
    
![image](https://github.com/user-attachments/assets/61d8c183-8314-48eb-8baa-d40fd6e1af7b)
 ### The image shows a count plot of age groups. The most frequent age group is 21-25, followed by 16-20, 26-30, and 31-35.

![image](https://github.com/user-attachments/assets/01d56105-3601-4d45-9680-4262e74f55e4)
### The bar chart showing the distribution of gender.There are two genders represented: Male and Female.
* Male is the more frequent gender in the dataset.
* Female is less frequent compared to Male.

![image](https://github.com/user-attachments/assets/63f71ed7-569c-4c33-89bd-23ac2eb8661d)
![image](https://github.com/user-attachments/assets/7907da9c-45ea-471d-9c2d-a2e0cb2308cb)
### The chart shows how often people use their mobile phones for education. Most people use it sometimes, while fewer people use it frequently or never.

![image](https://github.com/user-attachments/assets/a32b8736-41a4-4e14-bb5f-e8e462d72159)
### The bar chart depicts the count of mobile operating systems.

* Android: The taller bar represents that there are more devices using the Android operating system in the dataset.
* iOS: The shorter bar indicates a smaller number of devices using the iOS operating system compared to Android

![image](https://github.com/user-attachments/assets/43d7caef-64fd-4bbd-a7a6-4a9db925585e)

### The chart shows the frequency of different mobile phone activities.
"All of these" is the most common activity.
Other activities like messaging, social media, and web browsing are less frequent.
(all of these means :- Messaging,Social Media,web-browsing)

![image](https://github.com/user-attachments/assets/95d2865d-a594-4943-8580-b06de257efc3)
### The bar plot shows responses to whether an activity is helpful for studying:

* "No" Bar: Fewer than 10 respondents.
* "Yes" Bar: Around 85 respondents, indicating most find the activity helpful for studying.


The plot highlights a significant preference for "Yes" over "No".

![image](https://github.com/user-attachments/assets/d45b6536-8799-4cf7-99c9-cd92f846c3fc)

### The bar plot shows the usage distribution of different educational apps:

* Educational Videos: Most popular, with over 50 users.
* Language: Used by about 10 users.
*Productivity Tools: Used by about 10 users.
* Study Planner: Used by about 10 users.


Educational videos are significantly more popular compared to other app categories.


![image](https://github.com/user-attachments/assets/e025b28e-aad0-413c-bede-72ad619403d9)

### The chart shows the distribution of daily mobile phone usage.
* Most users spend 4-6 hours on their phones daily.
* A significant portion uses their phones for 2-4 hours.
* Fewer users spend less than 2 hours or more than 6 hours.
  

![image](https://github.com/user-attachments/assets/c462e085-d441-43d9-971d-c7010fe125d5)

### The chart shows the distribution of responses to "Performance impact".
* "Agree" has the highest count, indicating most people agree with the statement.
* "Disagree" and "Strongly disagree" have lower counts.
* "Neutral" and "Strongly agree" have moderate counts.

![image](https://github.com/user-attachments/assets/48df13c3-b50e-4f8b-af6b-21590750fe42)
### The chart shows how often mobile phone usage is distracting in different situations.

* Mobile phones are most distracting "While Studying".
* They are also quite distracting "During Class Lectures".
* Distractions are lower during "Exams" and when phones are "Not Distracting".


![image](https://github.com/user-attachments/assets/4a93da9a-a55c-46b0-8f29-99ebd1f180d8)

### The chart shows the distribution of attention span.
* More people have a good attention span.
* Fewer people have a poor attention span.

![image](https://github.com/user-attachments/assets/3e393abc-13b8-4f8a-a9d3-4c41f311a25a)
### The chart shows Useful feature of mobile phone
* Internet access is the primary feature used on mobile phones, followed by camera, note-taking apps, and calculators.

![image](https://github.com/user-attachments/assets/e002dc73-7899-4b0e-9c2d-725d75f8d587)
### The chart shows Count Plot of Health Risks
* Yes: Highest number of respondents reported health risks.
* Only Partially: Moderate number of respondents reported partial health risks.
* No: Lowest number of respondents reported no health risks.

![image](https://github.com/user-attachments/assets/90bb00bb-6b65-49df-b97e-743c307ed774)
### The chart shows Count Plot of Beneficial Subject
* Research is the most beneficial subject among respondents.
* Browsing Material is the second most beneficial subject.
* Accounting is the least beneficial subject based on the provided data.

![image](https://github.com/user-attachments/assets/05adf848-0784-4d80-803a-2c63df5a1f42)

### The chart shows Usage Symptoms
* Most common: Headache, Sleep disturbance, All of these.
* Less common: Anxiety or Stress.
* Least common: Headache; Sleep disturbance; Anxiety or Stress; All of these

![image](https://github.com/user-attachments/assets/970033c9-214c-4b18-b206-a5e03405648f)

### The chart shows Symptom Frequency
* Most respondents experience symptoms sometimes.
* A significant portion experiences symptoms rarely.
* Fewer respondents experience symptoms never or frequently.
  
![image](https://github.com/user-attachments/assets/9f37880b-506c-4199-9c40-bab1200f2b09)

### The chart shows Health Precautions
* Limiting Screen Time is the most common health precaution.
* None of Above is the second most common response.
* Taking a break during prolonged use is the third most common response.
* Using Blue light filter is the least common health precaution.
  
![image](https://github.com/user-attachments/assets/7d3f31b8-2280-4e66-beb6-8cc673110b77)

### Count Plot of Health Rating
* Most respondents rated their health as Good.
* A significant number rated their health as Excellent.
* Fewer respondents rated their health as Fair or Poor.
* There are very few respondents in the combined categories like Excellent; Good and Excellent; Good; Fair; Poor


## 3.Symptom frequency vs. usage symptoms
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
### 4.Contingency table and Chi-square test

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
 ### Contingency Table Analysis
* The heatmap presents a contingency table showing the relationship between mobile phone activities and their perceived impact on performance.

### Key Observations:
* Most respondents who engage in "All of these" activities agree that it impacts their performance.
* Messaging seems to have the least impact on performance.
* Social Media activities show a mixed impact, with a higher number of respondents agreeing on a negative impact.
* There are relatively few responses for the combined activities and their impact.

### Overall:
*The table suggests a potential correlation between certain mobile phone activities and their perceived impact on performance, with "All of these" activities showing the strongest association with a negative impact.

## 5.Health precautions vs. usage symptoms
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


## 6. Data Encoding and Splitting
### Encoding Categorical Variables
To prepare the data for modeling, we need to convert categorical variables into a numerical format. We use Label Encoding to achieve this
### Feature Selection and Train/Test Split
We separate the features and the target variable, then split the data into training and testing sets to avoid data leakage and ensure proper model evaluation



## 7.Model Training and Hyperparameter Tuning
We performed hyperparameter tuning on a Random Forest Classifier using GridSearchCV to identify the best combination of parameters for our model.

### Hyperparameter Grid:

* n_estimators: [50, 100, 200]
* max_depth: [None, 10, 20, 30]
* min_samples_split: [2, 5, 10]
* min_samples_leaf: [1, 2, 4]


### Best Parameters:
After evaluating 540 different combinations, the best parameters found were:

* max_depth: None
* min_samples_leaf: 4
* min_samples_split: 2
* n_estimators: 100
These parameters provided the most optimal performance for our dataset.

## 8.Feature Importance
![image](https://github.com/user-attachments/assets/1f59966d-bea9-4ab2-bebf-4a0244e57bd9)
The plot above shows the relative importance of each feature in the model. Features such as 'Symptom frequency', 'Health precautions', and 'Useful features' have the highest importance, indicating their significant impact on the model's performance. Other features like 'Mobile Phone' and 'Helpful for studying' have the least importance.

## 9. Model Evaluation

### Confusion matrix
![image](https://github.com/user-attachments/assets/dad972bd-96a1-46db-9a32-24be8add4b31)

### Overfitting and underfitting assessment
* Training Accuracy: 0.625
* Test Accuracy: 0.3157894736842105
* The model might be overfitting. Consider using techniques like cross-validation, pruning, or regularization.
  
### Sample predictions
* Sample Predictions: [2 0 0 0 2]
Actual Values: 41    2
23    2
57    0
97    0
0     0
Name: Performance impact, dtype: int64

### Decision Tree Visualization
![image](https://github.com/user-attachments/assets/aab1febc-98c1-4b5e-b5c4-bf7a3d94cbf8)



# Conclusion
* The analysis aimed to understand the impact of various factors on performance. Factors such as age, gender, mobile phone usage patterns, health conditions, and study habits were considered.

* Preliminary findings suggest that several factors contribute to performance impact. These include mobile phone activities, daily usage, health conditions, and symptoms. 
