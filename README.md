# 2024_ia651_RAGHU-_SAI_NATH_REDDY
# **Mobile Usage and Health Impact Analysis**
---
## objective :
The primary goal is to investigate the relationship between students' mobile phone usage and their academic performance, health, and overall well-being. We aim to identify potential correlations and patterns that can shed light on the impact of mobile phone usage on these factors.
---
## Loading Of Packages 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
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
![image](https://github.com/user-attachments/assets/01d56105-3601-4d45-9680-4262e74f55e4)
![image](https://github.com/user-attachments/assets/63f71ed7-569c-4c33-89bd-23ac2eb8661d)
![image](https://github.com/user-attachments/assets/7907da9c-45ea-471d-9c2d-a2e0cb2308cb)
![image](https://github.com/user-attachments/assets/a32b8736-41a4-4e14-bb5f-e8e462d72159)
![image](https://github.com/user-attachments/assets/43d7caef-64fd-4bbd-a7a6-4a9db925585e)
![image](https://github.com/user-attachments/assets/95d2865d-a594-4943-8580-b06de257efc3)
![image](https://github.com/user-attachments/assets/d45b6536-8799-4cf7-99c9-cd92f846c3fc)
![image](https://github.com/user-attachments/assets/e025b28e-aad0-413c-bede-72ad619403d9)
![image](https://github.com/user-attachments/assets/c462e085-d441-43d9-971d-c7010fe125d5)
![image](https://github.com/user-attachments/assets/48df13c3-b50e-4f8b-af6b-21590750fe42)
![image](https://github.com/user-attachments/assets/4a93da9a-a55c-46b0-8f29-99ebd1f180d8)
![image](https://github.com/user-attachments/assets/3e393abc-13b8-4f8a-a9d3-4c41f311a25a)
![image](https://github.com/user-attachments/assets/e002dc73-7899-4b0e-9c2d-725d75f8d587)
![image](https://github.com/user-attachments/assets/90bb00bb-6b65-49df-b97e-743c307ed774)
![image](https://github.com/user-attachments/assets/05adf848-0784-4d80-803a-2c63df5a1f42)
![image](https://github.com/user-attachments/assets/970033c9-214c-4b18-b206-a5e03405648f)
![image](https://github.com/user-attachments/assets/9f37880b-506c-4199-9c40-bab1200f2b09)
![image](https://github.com/user-attachments/assets/7d3f31b8-2280-4e66-beb6-8cc673110b77)


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
![image](https://github.com/user-attachments/assets/04169225-5e6a-45ec-ac88-e64f2db04515)

# Visualize the contingency table using a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(contingency_table, annot=True, cmap="YlGnBu", fmt='d')
plt.title('Contingency Table of Mobile Phone Activities and Performance Impact')
plt.show()
![image](https://github.com/user-attachments/assets/5465ae4c-e658-4fad-8321-43162edb2357)

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
<style>#sk-container-id-3 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-3 {
  color: var(--sklearn-color-text);
}

#sk-container-id-3 pre {
  padding: 0;
}

#sk-container-id-3 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-3 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-3 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-3 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-3 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-3 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-3 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-3 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-3 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-3 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-3 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-3 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-3 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-3 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-3 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-3 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-3 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-3 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-3 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-3 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-3 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-3 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-3 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-3 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-3 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-3 div.sk-label label.sk-toggleable__label,
#sk-container-id-3 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-3 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-3 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-3 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-3 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-3 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-3 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-3 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-3 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-3 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-3 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-3 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-3 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-3" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>RandomForestClassifier(random_state=42)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" checked><label for="sk-estimator-id-3" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;RandomForestClassifier<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.RandomForestClassifier.html">?<span>Documentation for RandomForestClassifier</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>RandomForestClassifier(random_state=42)</pre></div> </div></div></div></div>
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
![Uploading image.png…]()

# Overfitting and underfitting assessment
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)
print(f'Training Accuracy: {train_accuracy}')
print(f'Test Accuracy: {test_accuracy}')

if train_accuracy > test_accuracy + 0.1:
    print("The model might be overfitting. Consider using techniques like cross-validation, pruning, or regularization.")
elif train_accuracy < test_accuracy:
    print("The model might be underfitting. Consider using a more complex model or adding more features.")
<img width="799" alt="Screenshot 2024-07-25 at 7 09 31 PM" src="https://github.com/user-attachments/assets/7c07758a-04e8-4a7e-a7a1-d2c1f289edae">

# Sample predictions
sample_predictions = model.predict(X_test[:5])
print(f'Sample Predictions: {sample_predictions}')
print(f'Actual Values: {y_test[:5].values}')
<img width="274" alt="Screenshot 2024-07-25 at 7 10 29 PM" src="https://github.com/user-attachments/assets/c6d8a26f-d48c-41ce-a458-4741180d5fab">

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




