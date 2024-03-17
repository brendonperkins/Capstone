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
- **Area Under the Curve (AUC):** AUC is a performance metric associated with the Receiver Operating Characteristic (ROC) Curves, used for binary classification tasks. It quantifies a model's ability to distinguish between the classes—positive and negative. The AUC score ranges from 0 to 1, where a score of 1 indicates perfect ability to discriminate between positive and negative classes, and a score of 0.5 implies performance no better than random chance. A higher AUC value is indicative of a more effective model in terms of discrimination capacity.
- **Average Precision (AP):** AP is a single-figure metric derived from the Precision-Recall (PR) Curves, assessing the performance of binary classifiers in distinguishing between positive and negative classes. It evaluates how well a model sustains high precision as recall increases, which is crucial in contexts where the positive class is more significant or of specific interest. High AP values suggest that the model effectively balances precision with high recall, indicating superior performance in identifying positive instances while minimizing false positives.
- **Confusion Matrix:** This tool offers a comprehensive visualization of a classifier's performance, presenting the counts of true positive, false positive, true negative, and false negative predictions. The matrix's diagonal elements reflect accurate predictions for each class, with higher values indicating better performance. Off-diagonal elements represent instances misclassified by the model, providing insight into the types of errors made.
- **Accuracy:** Defined as the proportion of true results (both true positives and true negatives) among the total number of cases examined. It gives a straightforward measure of how often the classifier makes the correct prediction, though it may not always provide a complete picture, especially in imbalanced datasets.
- **Precision:** Precision measures the accuracy of positive predictions. It is calculated as the ratio of true positive predictions to the total number of positive predictions (true positives plus false positives). High precision indicates a low rate of false positive errors.
- **Recall (Sensitivity):** Recall assesses the model's ability to identify all relevant instances within a dataset. It is the ratio of true positive predictions to the actual number of positives (true positives plus false negatives). High recall means the model is good at capturing a large proportion of positive cases.
- **F1 Score:** The F1 Score is a harmonic mean of precision and recall, offering a balance between the two. It is particularly useful when you need to take both false positives and false negatives into account. The F1 Score reaches its best value at 1 (perfect precision and recall) and worst at 0, serving as a single metric to assess a model's accuracy while considering both precision and recall.

**APPROACHES**

Given the skewed distribution of class labels in the target variable, it's crucial to adopt scoring strategies that mitigate this imbalance, ensuring the final model maintains robust predictive capabilities in the deployed state. Thus, various scoring metrics were explored to identify the most effective model configuration:

1. **Baseline Accuracy Scoring w/o Statistical Dataset Resampling:** This method was employed as a reference point. It's widely acknowledged that relying solely on accuracy for scoring with imbalanced datasets often do not yield the most reliable predictive models since a model could achieve high accuracy by merely predicting the majority class.
2. **Accuracy Scoring w/ Statistical Dataset Resampling:** This method used SMOTE resampling of the dataset to adjust the ratio between the target variable minority and majority classes. This method tested whether statistical resampling could counteract the inherent bias of the dataset using two cases: 1) resampling of the minority:majority class ratio to 0.75:1 and 2) resampling of the minority:majority class ratio to 1:1.
3. **Geometric Mean Scoring:** This approach used G-Mean = sqrt(TPR * (1-FPR)) as the scoring criteria without dataset adjustment for target variable class imbalance. The geometric mean offers a balanced performance metric that penalizes models favoring the majority class over the minority, proving less influenced by outliers. This metric is particularly advantageous for models trained on imbalanced datasets and is known to correlate with improvements in the F1 score.
4. **Informedness Scoring:** This approach used Informedness = TPR- FPR as the scoring without dataset adjustment for target variable class imbalance. This scoring metric integrates sensitivity and specificity into a unified performance measure, providing a comprehensive evaluation of a classifier's effectiveness. It is known to be especially relevant in scenarios where both types of classification errors have significant implications, such as medical diagnostics or fraud detection, this method is beneficial for models trained on imbalanced datasets, promoting enhanced F1 scores.
5. **F1 Score Scoring:** This approach used F1 = 2 * (precision * recall) / (precision + recall) as the scoring criteria without dataset adjustment for target variable class imbalance. This scoring combines precision and recall into a single measure, offering a thorough assessment of the model's performance across both classes. Particularly vital in contexts where both false positives and negatives are costly. The F1 score is invaluable for models trained on imbalanced datasets, encouraging superior overall performance.

**OVERALL RESULTS**

The following plots show the results for each approach desribed above. For each approach, the metrics for Accuracy, Precision, Recall, and F1 are presented using the standard threshold of 0.5 and the optimized threshold for each classifier. All the approaches resulted in trained models with High AUC/PR scores, implying that all the models did a great job of distinguishing between the positive and negative classes of the target variable. However, the results also show that some approaches are less desireable in situations where target variable imbalances are present.
1. Statistical resampling is shown not to be an effective approach in avoiding training bias when using Accuracy as a scoring metric. In both cases where statistical resampling was implemented, degraded AUC's were noticed without any improvement in the F1 scores when compared to other approaches.
2. In all cases, the Random Forest classifier consistently outperformed all other classifiers for AUC and F1 with only slightly degraded Accuracies.
3. The Random Forest F1-Scored model provides the best overall performance with AUC=0.91 and F1=0.70 indicating that it is the best model for distinguishing beteween the target variable's positive and negative classes for the imbalanced dataset. Its Accuracy=0.84 was also the highest among models not specifically trained using Accuracy as the scoring metric and only slightly smaller than the highest Accuracy=0.86.
4. The metrics for all classifiers were better using F1 scoring than all the alternatives.

![ROC CURVES](https://github.com/brendonperkins/Capstone/assets/48937916/c51b3373-b893-4bbe-85b7-c45e33125aa3)

**DETAILED RESULTS FOR F1-SCORED MODELS**

The first analysis focused on the full dataset using F1 scoring in conjunction with Precision-Recall Curves to maximize the F1 score for the best performing model. No target variable re-balancing was implemented.

![ALL - PR-CURVE](https://github.com/brendonperkins/Capstone/assets/48937916/be83c4c0-b04b-4108-ac33-92ce19f7e142)

- **BEST MODEL:** It should be notied that all the models perform well as predictors of the target variable given the input variables. However, based on the performance metrics provided for each model, the RandomForest model was identified to be the best choice for this classification task. Here's why:

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
