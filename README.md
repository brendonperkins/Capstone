# BUSINESS UNDERSTANDING

## Dataset:
Adult Census Income from Kaggle

https://www.kaggle.com/uciml/adult-census-income

This data was extracted from the 1994 Census bureau database by Ronny Kohavi and Barry Becker (Data Mining and Visualization, Silicon Graphics). The task is to predict whether income exceeds $50k per year based on the census data provided above. The Dataset consists of a list of records, each of which captures various features of an individual including income per year.

## Nature of Task:
Supervised Machine Learning > Binary Classification

## Dataset Features:

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

# APPLICATION OF THE CRISP-DM PROCESS
1. Data Exploration Phase
    - Determine raw counts of Target Class >$50 by Class (for Categorical Columns) or Bracketed Value (for Numeric Columns).
    - Determine Target Ratio (= Raw Count of >$50k / Raw Count <=$50K) by Class (for Categorical Columns) or Bracketed Value (for Numeric Columns).
    - Plot histograms for numeric columns.
    - Plot correlation matrix (heat map for Numeric columns).
2. Data Preparation Phase
    - Identify and Address Missing Values
    - Identify and Address Outlier Values
    - Identify and Address Skewed Histograms
    - Identify Opportunities for Class Simplification
    - Identify Opportunities for Numeric Encoding of Categorical Columns
    - Determine Independence of Numeric Variables (heat map)
    - Implement Scaling for numeric columns
    - Split dataset into Training and Test Sets
    - Identify and Address Class Imbalances in Target Variable for Training set
3. Modeling Phase
    - Use GridSearchCV to tune hyper-parameters
    - Generate ROC curves and optimize thresholds
    - Assess confusion matrices and associated metrics
4. Evaluation Phase
    - Implement ROC Curves and select model with best performance
      - Determine optimal threshold
      - Report confusion matrix and associated metrics
      - Report feature importances for model with best performance
5. Deployment Phase
6. Suggested Follow-Up

## Data Exploration and Preparation Phases

Observations:
- Native-Born individuals overwhelmingly dominate the dataset (See Plot 08), so Native-Born and Foreign-Born individuals are split into two separate datasets for separate modeling and analysis. This prevents the Native-Born records from biasing the drivers of Foreign-Born success, which are likely different for each group.
- The number of records having a Target income <=$50K (Majority Class) is substantially more than the number having >$50K (Minority Class). The dataset needs to be balanced with the target values so that the models do not become biased to the majority class (See Plot 09).
- Individuals >75 years of age are not adequately represented in the dataset (See Plot 10), so the records for these individuals were dropped.
- Individuals with >65 hours-per-week are not adequately represented in the dataset (See Plot 11), so the records for these individuals were dropped.
- The “capital gain” and “capital loss” columns are sparsely populated with non-zero values, so they will be dropped.
- There are missing values in three of the dataset columns (workclass: 5.7%, occupation: 5.7%, and native country: 1.7%). Due to the sparseness of the missing values, records with missing values are dropped.
- Histogram plots of the numeric variables show they all have skewed distributions, so they will be replaced with log-transformed values. See Plots 22 - 24.
- There are no significant correlations between numeric variables (See Plot 25). Therefore, they can be treated as independent predictors of the Target variable.
- All classes representing less than a high school diploma in the education column have similar target ratios (See Plot 13), indicating that the differentiation of education below a high school diploma adds no insight into the drivers of income. Therefore, all classes representing education levels of less than a high school diploma were aggregated into a single class.
- The “education-num” column is duplicative of the “education” column, and is, therefore, dropped from the dataset.
- All numeric columns were scaled.
- All categorical columns were encoded using OneHotEncoding with column dropping enabled for columns with binary classes.
- Datasets were split into two sets: Tarining (70%) and Test (30%).
- The datasets were split into training and test sets using a 70%/30% split.
- The Synthetic Minority Over-sampling (SMOTE) method was then used to ensure a Minority/Majority ratio of 0.75 to 1 for the Target Variable classes in the Training set only. 

![raw counts copy](https://github.com/brendonperkins/Capstone/assets/48937916/347ae187-2254-4a04-993b-7b91c9ae3d1a)

## Modeling Phase
The following set of models were developed and trained separately on the two datasets, the first for Native-Born individuals and the second for Foreign-Born individuals:

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
Model evaluation metrics are required to quantify model performance. The choice of evaluation metrics depends on a given machine learning task (such as classification, regression, ranking, clustering, topic modeling, among others). Some metrics, such as precision-recall, are useful for multiple tasks.

**METRICS**
- **Area Under Curve (AUC):** Is a performance metric for measuring the ability of a binary classifier to discriminate between positive and negative classes.
- **Confusion Matrix:** Provides a more detailed breakdown of correct and incorrect classifications for each class.The diagonal elements represent the number of points for which the predicted label is equal to the true label, while anything off the diagonal was mislabeled by the classifier. Therefore, the higher the diagonal values of the confusion matrix the better, indicating many correct predictions.
- **Accuracy:** Is a common evaluation metric for classification problems. It’s the number of correct predictions made as a ratio of all predictions made. We use sklearn module to compute the accuracy of a classification task
the score.
- **Precision:** Is the number of correct positive results divided by the total predicted positive observations.
- **Recall:** Is the number of correct positive results divided by the number of all relevant samples (total actual
- **F--score:** Is a measure of a test’s accuracy that considers both the precision and the recall of the test to compute positives).

**MODELS**
- **LOGISTIC REGRESSION:** 
![ALL - LR](https://github.com/brendonperkins/Capstone/assets/48937916/6146828e-c413-4073-9c7b-c429aa51d667)

- **K-NEAREST NEIGHBORS:**
![ALL - KN](https://github.com/brendonperkins/Capstone/assets/48937916/de759f78-bf23-4af5-a027-ea6bb4758471)

- **DECISION TREE:**
![ALL - DT](https://github.com/brendonperkins/Capstone/assets/48937916/dc633475-9dd6-4334-8722-853e3bf8bad6)

- **RANDOM FOREST:** 
![ALL - RF](https://github.com/brendonperkins/Capstone/assets/48937916/d8aae911-de94-4fa1-b048-72697a0cb354)

- **NEURAL NETWORK:** 
![ALL - NN](https://github.com/brendonperkins/Capstone/assets/48937916/0415b6cd-5d10-4dda-be4d-861cdc75000e)

- **SUPPORT VECTOR MACHINE:** 
![ALL - SVM](https://github.com/brendonperkins/Capstone/assets/48937916/4b965319-8697-4e47-a7ca-9b22a69671d4)

### Results for Native-Born Dataset
- Best Model: 
  - Logistic Regression (C=1, penalty=L1)
  - AUC = 0.89
  - Optimal Threshold: 0.39
  - Confusion Matrix: [[3998 1265][ 263 1506]]
    - Accuracy = 0.78 
    - Precision = 0.54
    - Recall = 0.85
    - F1 Score = 0.66
![native](https://github.com/brendonperkins/Capstone/assets/48937916/4f4c762a-2b3b-4a60-9fc2-9790c80bf2e7)
![IMPORTANCE - NATIVE](https://github.com/brendonperkins/Capstone/assets/48937916/475ab138-c9d0-4a16-8ba2-a63335cd8f22)
### Results for Foreign-Born Dataset

- Best Model: 
  - Random Forest (max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=100)
  - AUC = 0.89
  - Optimal Threshold: 0.19
  - Confusion Matrix: [[417 122][ 15 119]]
    - Accuracy = 0.80
    - Precision = 0.49
    - Recall = 0.89
    - F1 Score = 0.63
![foreign](https://github.com/brendonperkins/Capstone/assets/48937916/924c837d-e4fc-40a8-aac4-0cb676b07294)
![IMPORTANCE - FOREIGN](https://github.com/brendonperkins/Capstone/assets/48937916/9f781495-091c-4a46-aa75-965fecc15595)
## Deployment Phase
- Native- and Foreign-Born Models are valid only for:
  - Ages 18 to 75 years of age.
  - Hours-per-Week of 15 to 65
  - Datasets that have been prepared similar to the training set

# Conclusion
- Both datasets identified Occupation, Education, and Relationship within the top 5 most important parameters for predicting the likelihood of achieving incomes >$50K. For the Foreign-Born dataset, native-country is also an important factor.
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

