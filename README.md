# Business Understanding

**Dataset:** Adult Census Income from Kaggle

**URL:** https://www.kaggle.com/uciml/adult-census-income

**Description:** This data was extracted from the 1994 Census bureau database by Ronny Kohavi and Barry Becker (Data Mining and Visualization, Silicon Graphics). The task is to predict whether income exceeds $50k per year based on the census data provided above. The Dataset consists of a list of records, each of which captures various features of an individual including income per year.

**Nature of Task:** Supervised Machine Learning > Binary Classification

**Objective:** Develop models using multiple classifiers that predict whether an individual's annual income exceeds $50K based on 14 input paramters provided in the dataset. Compare and contrast the models to select the best model and extract feature importances for the best model. 

# Summarized Results
- The Random Forest model identified Marital Status, Captial-Gains, Education, Relationship, Occupation, and Age as the top 6 most important parameters for predicting the likelihood of achieving incomes >$50K. The least important parameters included race,native-country, gender, and working class as the least significant.
- Entrepreneurs are more likely to exceed $50K income threshold, followed by individuals working in Federal, State, and Local governments, and individuals working in the private sector (See Plot 12).
- Higher levels of education correspond with higher probabilities of exceeding the $50K income threshold (See Plot 13).
- Married individuals are far likelier than single individuals to exceed the $50K income threshold (See Plot 14).
- Whites are far likelier than other races to exceed the $50K income threshold. Note: Asian-Pacific Islanders class exceeds Whites, but could be an anomalous result due to the sparse representation of this class in the dataset (See Plot 17).
- Males are far likelier to exceed the $50K income threshold (See Plot 18).
- Foreign-Born individuals originating from the Caribbean and Central/South America are the least likely to exceed the $50K income threshold (See Plot 19).
- Foreign-Born from Canada, Europe, and Asia are the most likely to exceed the $50K income threshold (See Plot 19).
- Likelihood of exceeding $50K income threshold increases with age up to 55 years and decreases thereafter (See Plot 20).
- Likelihood of exceeding $50K threshold increases with hours per week up to 60 and decreases thereafter before leveling off at 80 (See Plot 21).

# Suggested Follow-Up
Look at the underlying reasons for the lower likelihood of Foreign-Born individuals from Caribbean and Central/South American countries exceeding the $50K income threshold. Determine if underlying causes are 1) the result of socio-economic policies in native countries of origin, or 2) the result of a direct land bridge to the United States, making it a more accessible destination for individuals of all socio-economic classes.

# Dataset Features

The dataset contains 43,957 entries.

| #   | Type        | Variable       | Values        |
| --- | ----------- | -------------- | ------------- |
|  1. | categorical | income         | >50K, <=50K   |
|  2. | numeric     | age            | continuous    |
|  3. | categorical | workclass      | Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked |
|  4. | numeric     | fnlwgt         | continuous    |
|  5. | categorical | education      | Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool |
|  6. |  numeric    | education-num  | continuous |
|  7. | categorical | marital-status | Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse |
|  8. | categorical | occupation     | Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces |
|  9. | categorical | relationship   | Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried |
| 10. | categorical | race           | White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black |
| 11. | categorical | sex            | Female, Male |
| 12. | numeric     | capital-gain   | continuous |
| 13. | numeric     | capital-loss   | continuous |
| 14. | numeric     | hours-per-week | continuous |
| 15. | categorical | native-country | United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands |

# Application of the CRISP-DM Process
1. **Data Exploration Phase**
    - Determine raw counts of Target Class >$50 by Class (for Categorical Columns) or Bracketed Value (for Numeric Columns).
    - Determine Target Ratio (= Raw Count of >$50k / Raw Count <=$50K) by Class (for Categorical Columns) or Bracketed Value (for Numeric Columns).
    - Plot histograms for numeric columns.
    - Plot correlation matrix (heat map for Numeric columns).
2. **Data Preparation Phase**
    - Identify and Address Missing Values
    - Identify and Address Outlier Values
    - Identify and Address Skewed Histograms
    - Identify Opportunities for Class Simplification
    - Identify Opportunities for Numeric Encoding of Categorical Columns
    - Determine Independence of Numeric Variables (heat map)
    - Implement Scaling for numeric columns
    - Split dataset into Training and Test Sets
    - Identify and Address Class Imbalances in Target Variable for Training set
3. **Modeling Phase**
    - Use GridSearchCV to tune hyper-parameters
    - Generate ROC curves and optimize thresholds
    - Assess confusion matrices and associated metrics
4. **Evaluation Phase**
    - Implement ROC Curves and select model with best performance
      - Determine optimal threshold
      - Report confusion matrix and associated metrics
      - Report feature importances for model with best performance
5. **Deployment Phase**

## Data Exploration and Preparation Phases

**Observations:**
- The dataset is predominantly composed of Native-Born individuals, as illustrated in Plot 08. It's important to consider that the factors influencing income levels may differ between Foreign-Born and Native-Born individuals. Given the imbalance bewteen these two classes, a second dataset is prepared for distinct modeling and analysis comprising only of the Foreign-Born individuals. This approach ensures that the analysis of factors influencing the success of Foreign-Born individuals is not skewed by the characteristics of the majority Native-Born class.
- The number of records having a target income <=$50K (Majority Class) is substantially more than the number having >$50K (Minority Class). To address this imbalance, two modeling approaches were explored: 1) the models were scored using F1 as the metric and Precision-Recall curves to select an optimal threshold for generating confusion matrices and 2) the dataset was rebalanced before using Accuracy as the scoring metric and ROC curves to select the optimal threshold for generating confusion matrices.
- Individuals >75 years of age are not adequately represented in the dataset (See Plot 10), so the records for these individuals were dropped.
- Individuals with >65 hours-per-week are not adequately represented in the dataset (See Plot 11), so the records for these individuals were dropped.
- The “capital loss” column is folded into the “capital gain” column.
- There are missing values in three of the dataset columns (workclass: 5.7%, occupation: 5.7%, and native country: 1.7%). Due to the sparseness of the missing values, records with missing values are dropped.
- Histogram plots of the numeric variables show they all have skewed distributions, so consider replacing them with their log-transformed values. See Plots 22 - 24.
- There are no significant correlations between numeric variables (See Plot 25). Therefore, they will be treated as independent predictors of the Target variable.
- All classes representing less than a high school diploma in the education column have similar target ratios (See Plot 13), indicating that the differentiation of education below a high school diploma adds no insight into the drivers of the Target variable. Therefore, all classes representing education levels of less than a high school diploma are aggregated into a single class.
- The “education-num” column is duplicative of the “education” column, and is, therefore, dropped from the dataset.
- All numeric columns were scaled to mean = 0 and standard devaiation = 1.
- All categorical columns are encoded using OneHotEncoding with column dropping enabled for columns with binary classes.
- After completion of the data cleaning steps, the dataset was reduced to 38,524 entries from its original 43,957 entries. Of those entries remaining, 9,615 entries represent individuals with incomes ">$50K" while 28,909 entries represent individuals with incomes "<=$50K".

![raw counts copy](https://github.com/brendonperkins/Capstone/assets/48937916/347ae187-2254-4a04-993b-7b91c9ae3d1a)

## Modeling Phase

A series of classifiers was independently constructed and trained on multiple datasets with each partitioned into a 80% training and 20% testing split. The datasets used were the original cleaned dataset and a separate dataset prepared from the cleaned dataset to include only Foreign-Born individuals. The GridSearchCV tool was employed to fine-tune each classifier based on a predefined set of hyper-parameters, ensuring optimal performance. Due to the imbalanced nature of the target variable, the F1 score — a harmonic mean of precision and recall — was chosen as the guiding metric for training. Precision-Recall curves were subsequently plotted for each classifier, with the area under each curve quantified as the Average Precision, reflecting the model's performance across various threshold levels.

These Precision-Recall curves were instrumental in determining the most advantageous threshold that maximizes the F1 score, thereby achieving a harmonious balance between Precision (the model's ability to identify only relevant instances) and Recall (the model's ability to identify all relevant instances). Utilizing this optimal threshold, I then computed and compared the confusion matrix and related metrics — namely the F1 Score, Accuracy, Precision, and Recall — at both the default threshold of 0.5 and the optimal threshold. This comparison illustrates the enhancements achieved through the application of the optimal threshold.

The classifier that demonstrated the highest Average Precision alongside the highest F1 score was subsequently identified as the best-performing model, signifying its superiority in balancing precision and recall and effectively addressing the challenge posed by the class imbalance in the target variable.

- LOGISTIC REGRESSION: A model for binary classification, predicting the probability of a default class instance using a logistic function. 
  - Hyper-Params:
  - C: 0.001, 0.01, 0.1, 1, 10, 100, 1000
  - penalty: l1, l2

- K-NEAREST NEIGHBORS: A non-parametric method that classifies a majority class instance among its closest K neighbors.
  - Hyper-Params: 
  - n_neighbors: 3, 5, 7, 10
  - metric: euclidean, Manhattan
  - weights: uniform, distance

- DECISION TREE: A tree-like model of decisions and consequences, useful for modeling decisions with conditional control statements.
  - Hyper-Params:
  - max_depth: None, 5, 10, 15, 20, 30
  - min_samples_leaf: 1, 2, 4
  - min_samples_split: 2, 5, 10 

- RANDOM FOREST: An ensemble learning method that incorporates "bagging," or bootstrap aggregating, to improve stability and accuracy in machine learning models. 
  - Hyper-Params: n_estimators: 10, 50, 100
  - max_depth: None, 5, 10, 15, 20
  - min_samples_leaf: 1, 2, 4
  - min_samples_split: 2, 5, 10

- SUPPORT VECTOR MACHINE: Maximizes the margin between the datapoints of different classes.
  - Hyper-Params: C: 0.1, 1, 10, 100
  - kernel: linear, rbf; 
  - gamma: scale, auto

- NEURAL NETWORK: Algorithms that mimic brain function to discern data relationships, capable of adapting to changing inputs to generate optimal results.
  - Hyper-Params: epochs: 10, 20
  - batch_size: 16, 32
  - layers: [64, 64], [64, 64, 32]
  - optimizer: adam, rmsprop
  - dropout_rate: 0.0, 0.2

## Evaluation Phase
Model evaluation metrics are required to quantify model performance. The choice of evaluation metrics depends on a given machine learning task (e.g. classification, regression, clustering). Some metrics, such as precision-recall, are useful for multiple tasks. The following metrics were used to evaluate the models.

**METRICS**
- **Area Under Curve (AUC):** Is a performance metric for measuring the ability of a binary classifier to discriminate between positive and negative classes.
- **Confusion Matrix:** Provides a more detailed breakdown of correct and incorrect classifications for each class.The diagonal elements represent the number of points for which the predicted label is equal to the true label, while anything off the diagonal was mislabeled by the classifier. Therefore, the higher the diagonal values of the confusion matrix the better, indicating many correct predictions.
- **Accuracy:** Is a common evaluation metric for classification problems. It’s the number of correct predictions made as a ratio of all predictions made. We use sklearn module to compute the accuracy of a classification task
the score.
- **Precision:** Is the number of correct positive results divided by the total predicted positive observations.
- **Recall:** Is the number of correct positive results divided by the number of all relevant samples (total actual
- **F-score:** Is a measure of a test’s accuracy that considers both the precision and the recall of the test to compute positives).

**APPROACHES**

Given the skewed distribution of class labels in the target variable, it's crucial to adopt scoring strategies that mitigate this imbalance, ensuring the final model maintains robust predictive capabilities. Thus, various scoring metrics were explored to identify the most effective model configuration:

1. **Baseline Accuracy Scoring without modifying the dataset:** to address class imbalance. This method was employed as a reference point. It's widely acknowledged that relying solely on accuracy for scoring in imbalanced datasets may not yield the most reliable predictive models since a model could achieve high accuracy by merely predicting the majority class.
2. **Accuracy Scoring with Statistical Resampling:** to adjust the ratio between the minority and majority classes to 0.75:1. This method tested whether statistical resampling could counteract the inherent bias of the dataset.
3. **Enhanced Accuracy Scoring with Equal Resampling:** aiming for a 1:1 balance between classes through dataset manipulation. This variant further examined the impact of the resampling level on model performance.
4. **Geometric Mean Scoring (Geometric Mean = sqrt(TPR * (1-FPR))):** without dataset adjustment for class imbalance. The geometric mean offers a balanced performance metric that penalizes models favoring the majority class over the minority, proving less influenced by outliers. This metric is particularly advantageous for models trained on imbalanced datasets and is known to correlate with improvements in the F1 score.
5. **Informedness Scoring (Informedness = TPR - FPR) without dataset resampling:** This metric integrates sensitivity and specificity into a unified performance measure, providing a comprehensive evaluation of a classifier's effectiveness. Especially relevant in scenarios where both types of classification errors have significant implications, such as medical diagnostics or fraud detection, this method is beneficial for models trained on imbalanced datasets, promoting enhanced F1 scores.
6. **F1 Score Scoring (F1 = 2 * (precision * recall) / (precision + recall)):** without dataset modifications. This scoring combines precision and recall into a single measure, offering a thorough assessment of the model's performance across both classes. Particularly vital in contexts where both false positives and negatives are costly, the F1 score is invaluable for models trained on imbalanced datasets, encouraging superior overall performance.

![ROC CURVES](https://github.com/brendonperkins/Capstone/assets/48937916/c51b3373-b893-4bbe-85b7-c45e33125aa3)

**RESULTS - ANALYSIS #1**

The first analysis focused on the full dataset using F1 scoring in conjunction with Precision-Recall Curves to maximize the F1 score for the best performing model. No target variable re-balancing was implemented.

![ALL - PR-CURVE](https://github.com/brendonperkins/Capstone/assets/48937916/be83c4c0-b04b-4108-ac33-92ce19f7e142)

- **BEST MODEL:** It should be notied that all the models perform well as predictors of the target variable given the input variables. However, based on the performance metrics provided for each model, the RandomForest model was identified to be the best choice for this classification task. Here's why:

    - **Highest Average Precision (Area):** The has the hishest Average Precision (identified as "Area" on the plot) among the alternatives with a value of 0.79, suggesting that it captures the best F1 score, the harmonic mean of precision and recall. F1 is, indeed, the highest for RandomForest (0.681532) at the default threshold, suggesting it provides the best balance between precision and recall among the models tested.

    - **Highest F1 Score:** The F1 score, which is the harmonic mean of precision and recall, is highest for RandomForest (0.681532) at the default threshold, suggesting it has the best balance between precision and recall among the models tested.

    - **Highest Overall Accuracy:** RandomForest achieved the highest accuracy (0.855419) at the default threshold, indicating it correctly classified a higher percentage of instances compared to other models.

    - **Balance between Precision and Recall:** It also demonstrated a strong balance between precision (0.768536) and recall (0.612224) at the default threshold, indicating a good balance between the rate of true positive predictions and the model's ability to capture the positive class.

- **LOGISTIC REGRESSION:** The LogisticRegression model displayed a solid performance with an accuracy of 0.840493, precision of 0.728081, and recall of 0.588598, leading to an F1 score of 0.650951. It found its best parameters to be {'C': 1.0, 'penalty': 'l1'}, achieving this with a relatively quick training duration of 7.07 seconds. The model demonstrated a good balance between precision and recall, making it a competent choice for binary classification tasks, especially considering its efficiency in training time.
![ALL - LR](https://github.com/brendonperkins/Capstone/assets/48937916/eb03d853-81b1-4aea-91c7-bae54781e5b3)

- **K-NEAREST NEIGHBORS:** KNeighbors, with its best parameters set to {'metric': 'euclidean', 'n_neighbors': 10, 'weights': 'uniform'}, achieved an accuracy of 0.824659. The model had a precision of 0.659018 and a recall of 0.634309, resulting in an F1 score of 0.646428. Although it required a longer training time of 53.92 seconds, its performance indicates a moderate balance between identifying the positive class and minimizing false positives, though it lagged slightly behind other models in overall accuracy.
![ALL - KN](https://github.com/brendonperkins/Capstone/assets/48937916/8196125b-d7d4-49b8-b614-a205adbcc168)

- **DECISION TREE:** The DecisionTree model, optimized with {'max_depth': 10, 'min_samples_leaf': 4, 'min_samples_split': 2}, showed strong results with an accuracy of 0.845165, a precision of 0.743855, and a recall of 0.590652, which led to an F1 score of 0.658460. Its training was notably efficient, taking only 6.53 seconds, suggesting that DecisionTree is a fast and fairly accurate model for handling binary classification, though it faces limitations in balancing precision and recall as effectively as some other models.
![ALL - DT](https://github.com/brendonperkins/Capstone/assets/48937916/58fdb454-f8f6-413d-8d08-7807e12bfc15)

- **RANDOM FOREST:** Demonstrating the best overall performance, the RandomForest model, with parameters {'max_depth': None, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 100}, achieved the highest accuracy of 0.855419 among the models tested. It showed a precision of 0.768536 and a recall of 0.612224, leading to the highest F1 score of 0.681532. Despite a longer training duration of 98.75 seconds, its ability to maintain a high accuracy while also achieving the best balance between precision and recall marks it as the top contender.
![ALL - RF](https://github.com/brendonperkins/Capstone/assets/48937916/5f133504-5b2a-4e33-b73b-3f4a2007dc1a)

- **NEURAL NETWORK:** The NeuralNetwork, optimized with {'batch_size': 32, 'dropout_rate': 0.2, 'epochs': 20, 'layers': [64, 64, 32], 'optimizer': 'adam'}, recorded an accuracy of 0.840493. It achieved a precision of 0.764749 and a recall of 0.532614, culminating in an F1 score of 0.627914. While its training duration was the longest at 471.90 seconds, the model showed strong potential in minimizing false positives but at the expense of a lower recall rate, indicating a challenge in identifying all positive cases effectively.
![ALL - NN](https://github.com/brendonperkins/Capstone/assets/48937916/dff44c27-ac4a-4dfd-99c8-4dba45791063)

- **SUPPORT VECTOR MACHINE:** SVM found its best performance with parameters {'C': 100, 'gamma': 'auto', 'kernel': 'rbf'}, reaching an accuracy of 0.842310. The model had a high precision of 0.806020 but a lower recall of 0.495121, resulting in an F1 score of 0.613427. Its training took the longest time of 549.31 seconds, suggesting that while SVM is very capable of identifying positive cases with high confidence, it struggles more with recall, making it less efficient at capturing the majority of the positive class.
![ALL - SVM](https://github.com/brendonperkins/Capstone/assets/48937916/ebbb1532-81c3-4a43-a4c6-f4cf2987158f)


**RESULTS - ANALYSIS #2**

The second analysis focused on the full dataset using Accuracy as the scoring criteria in conjunction with ROC Curves to maximize the Accuracy score for the best performing model. Target variable re-balancing was implemented using SMOTE to rebalance the target majority/minotiy classes to a 1:1 ratio.

![ALL - PR-CURVE](https://github.com/brendonperkins/Capstone/assets/48937916/be83c4c0-b04b-4108-ac33-92ce19f7e142)

- **BEST MODEL:** It should be notied that all the models perform well as predictors of the target variable given the input variables. However, based on the performance metrics provided for each model, the RandomForest model was identified to be the best choice for this classification task. Here's why:

    - **Highest Average Precision (Area):** The has the hishest Average Precision (identified as "Area" on the plot) among the alternatives with a value of 0.79, suggesting that it captures the best F1 score, the harmonic mean of precision and recall. F1 is, indeed, the highest for RandomForest (0.681532) at the default threshold, suggesting it provides the best balance between precision and recall among the models tested.

    - **Highest F1 Score:** The F1 score, which is the harmonic mean of precision and recall, is highest for RandomForest (0.681532) at the default threshold, suggesting it has the best balance between precision and recall among the models tested.

    - **Highest Overall Accuracy:** RandomForest achieved the highest accuracy (0.855419) at the default threshold, indicating it correctly classified a higher percentage of instances compared to other models.

    - **Balance between Precision and Recall:** It also demonstrated a strong balance between precision (0.768536) and recall (0.612224) at the default threshold, indicating a good balance between the rate of true positive predictions and the model's ability to capture the positive class.

- **LOGISTIC REGRESSION:** The LogisticRegression model displayed a solid performance with an accuracy of 0.840493, precision of 0.728081, and recall of 0.588598, leading to an F1 score of 0.650951. It found its best parameters to be {'C': 1.0, 'penalty': 'l1'}, achieving this with a relatively quick training duration of 7.07 seconds. The model demonstrated a good balance between precision and recall, making it a competent choice for binary classification tasks, especially considering its efficiency in training time.
![ALL - LR - ROC](https://github.com/brendonperkins/Capstone/assets/48937916/ecf72e35-eb36-4b92-83d3-92d756092f29)

- **K-NEAREST NEIGHBORS:** KNeighbors, with its best parameters set to {'metric': 'euclidean', 'n_neighbors': 10, 'weights': 'uniform'}, achieved an accuracy of 0.824659. The model had a precision of 0.659018 and a recall of 0.634309, resulting in an F1 score of 0.646428. Although it required a longer training time of 53.92 seconds, its performance indicates a moderate balance between identifying the positive class and minimizing false positives, though it lagged slightly behind other models in overall accuracy.
![ALL - KN - ROC](https://github.com/brendonperkins/Capstone/assets/48937916/95978b67-c418-42bc-9b4d-c3ede51ee38d)

- **DECISION TREE:** The DecisionTree model, optimized with {'max_depth': 10, 'min_samples_leaf': 4, 'min_samples_split': 2}, showed strong results with an accuracy of 0.845165, a precision of 0.743855, and a recall of 0.590652, which led to an F1 score of 0.658460. Its training was notably efficient, taking only 6.53 seconds, suggesting that DecisionTree is a fast and fairly accurate model for handling binary classification, though it faces limitations in balancing precision and recall as effectively as some other models.
![ALL - DT - ROC](https://github.com/brendonperkins/Capstone/assets/48937916/1928f788-9a89-46dd-87d9-dc152c6a1785)

- **RANDOM FOREST:** Demonstrating the best overall performance, the RandomForest model, with parameters {'max_depth': None, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 100}, achieved the highest accuracy of 0.855419 among the models tested. It showed a precision of 0.768536 and a recall of 0.612224, leading to the highest F1 score of 0.681532. Despite a longer training duration of 98.75 seconds, its ability to maintain a high accuracy while also achieving the best balance between precision and recall marks it as the top contender.
![ALL - RF - ROC](https://github.com/brendonperkins/Capstone/assets/48937916/fafadc56-547a-4b7c-b36c-5078b6abf385)

- **NEURAL NETWORK:** The NeuralNetwork, optimized with {'batch_size': 32, 'dropout_rate': 0.2, 'epochs': 20, 'layers': [64, 64, 32], 'optimizer': 'adam'}, recorded an accuracy of 0.840493. It achieved a precision of 0.764749 and a recall of 0.532614, culminating in an F1 score of 0.627914. While its training duration was the longest at 471.90 seconds, the model showed strong potential in minimizing false positives but at the expense of a lower recall rate, indicating a challenge in identifying all positive cases effectively.
![ALL - NN - ROC](https://github.com/brendonperkins/Capstone/assets/48937916/2c3bcd4d-6242-4465-900c-4703e3bed1a6)

- **SUPPORT VECTOR MACHINE:** SVM found its best performance with parameters {'C': 100, 'gamma': 'auto', 'kernel': 'rbf'}, reaching an accuracy of 0.842310. The model had a high precision of 0.806020 but a lower recall of 0.495121, resulting in an F1 score of 0.613427. Its training took the longest time of 549.31 seconds, suggesting that while SVM is very capable of identifying positive cases with high confidence, it struggles more with recall, making it less efficient at capturing the majority of the positive class.
