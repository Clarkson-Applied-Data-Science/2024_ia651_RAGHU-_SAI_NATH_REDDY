# 2024_ia651_RAGHU-_SAI_NATH_REDDY
# Impact of Mobile Phone Habits on Student Health and Performance

## Project Overview
This project investigates the relationship between students' mobile phone usage and their academic performance, health, and overall well-being. The goal is to identify potential correlations and patterns that shed light on the impact of mobile phone usage on these factors.

## Dataset Description
The dataset comprises information about students, their mobile phone usage, and its potential impact on their academic performance and health. Each row represents a student, and columns contain attributes such as demographics, mobile phone details, usage patterns, academic indicators, and health-related factors.

### Columns in the Dataset
- **Names:** Student names
- **Age:** Student ages (in years)
- **Gender:** Male/Female
- **Mobile Phone:** Do students own a mobile phone? (Yes/No)
- **Mobile Operating System:** Type of mobile operating system used (e.g. Android, iOS, Other)
- **Mobile phone use for education:** Do students use their mobile phone for educational purposes? (Sometimes/Frequently/Rarely)
- **Mobile phone activities:** List of mobile phone activities used for educational purposes (e.g. online research, educational apps, email, online learning platforms)
- **Helpful for studying:** Do students find mobile phone use helpful for studying? (Yes/No)
- **Educational Apps:** List of educational apps used
- **Daily usage:** Average daily time spent using mobile phone for educational purposes (in hours)
- **Performance impact:** How does mobile phone use impact academic performance? (Agree/Neutral/Strongly agree)
- **Usage distraction:** Does mobile phone use distract from studying? (During Exams/Not Distracting/During Class Lectures/While Studying)
- **Attention span:** Has mobile phone use affected attention span? (Yes/No)
- **Useful features:** What features of mobile phones are useful for learning? (e.g. Internet Access, Camera, Calculator, Notes Taking App)
- **Health Risks:** Are students aware of potential health risks associated with excessive mobile phone use? (Yes/No/Only Partially)
- **Beneficial subject:** Which subjects benefit most from mobile phone use? (e.g. Accounting, Browsing Material, Research)
- **Usage symptoms:** Are students experiencing any physical or mental symptoms related to mobile phone use? (e.g. Sleep disturbance, headaches, Anxiety or Stress, All of these)
- **Symptom frequency:** How often are symptoms experienced? (Sometimes/Never/Rarely/Frequently)
- **Health precautions:** Are students taking precautions to mitigate potential health risks? (Taking Break during prolonged use/Using Blue light filter/Limiting Screen Time/None of Above)
- **Health rating:** How would students rate their overall physical and mental health? (Excellent/Good/Fair/Poor)

## Steps to Follow
1. **Data Cleaning and Preparation:**
   - Handle missing values, outliers, and inconsistencies.
   - Encode categorical variables using dummy variables.

2. **Exploratory Data Analysis (EDA):**
   - Examine distributions, correlations, and relationships between variables.
   - Create visualizations (histograms, box plots, scatter plots, correlation matrices) to understand data characteristics.

3. **Feature Engineering:**
   - Normalize numerical features.
   - Create lag variables if predicting future values.
   - Ensure no variables in the X data exactly equal the y variable.

4. **Model Building:**
   - Linear Regression: Suitable for predicting a continuous numerical Y variable (e.g. academic performance score).
     - Evaluation metrics: Mean Squared Error (MSE), Mean Absolute Error (MAE), R-squared.
   - Logistic Regression: Suitable for predicting a binary categorical Y variable (e.g. pass/fail, high/low health rating).
     - Evaluation metrics: Accuracy, precision, recall, F1-score, ROC curve, AUC.

5. **Avoid Data Leakage:**
   - Apply undersampling, oversampling, SMOTE, scaling, and PCA only to training/validation data, not testing data.
   - Test the model using unbalanced test data to ensure it generalizes well.

6. **Model Evaluation:**
   - Use precision and recall to quantify results, especially for unbalanced test data.


