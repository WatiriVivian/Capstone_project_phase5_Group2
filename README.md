# Football predictive model
## Business Problem
Assist betting enthusiasts in making more informed decisions.
> The **Football Predictive Model** enhances football decision-making using data analytics and machine learning.
> It offers insights, predictions, and recommendations based on historical data and player statistics, benefiting clubs, coaches, and stakeholders, as well as providing accurate predictions for betting groups.

## Data understanding

This dataset was scraped from [website](https://fbref.com/en/comps/9/Premier-League-Stats):

There are **46 columns** and **4700**

It features three data types: 
 * 1 Integer
 * 31 Float
 * 14 Object columns
   
Columns include date, match details, team information, and various performance metrics.

Notably, the 'Attendance' column has around 29% null values, and the 'Notes' column is entirely null.

Fortunately, there are no duplicate records.

Also in this dataset, 26 unique Premier League teams have participated, each playing a varying number of matches ranging from 38 to 228 over multiple seasons.

#### Data dictionary

Date :  date the match was played

Time :  local time the match was played

Comp:  level in the country's league pyramid the league occupies

Round:  phase of competition

Day :  day of week

Venue:  Stadium where the game was played either away or home

Result:  Match outcome either a win,  draw or loss

GF: goals for

GA:  goals against

Opponent:  team playing against

XG:  expected goals not including penalty shootouts

XGA:  expected goals allowed

Poss:  percentage of passes attempted

Attendance:  attendance per game during the season for home matches

Captain: Team lead player

Formation : number of players in each row from defenders to forwards, not including the goalkeeper

Referee: Neutral player officiating the match

Match Report:  contains summary of all columns

Notes:  additional details on the scores i.e penalty goals and whether there was any additional time

Sh :  total shots not including penalty kicks

SoT :  shots on target not including penalty kicks

FK :  shots from free kicks resulting from fouls

Dist :  average distance in yards from goal of all shots taken

PK: Penalty kicks

PKatt : penalty kicks attempted

Season:  Playing period 

Team : team playing

## Data Preparation and cleaning
*Importing the necessary libraries into our notebook for analysis
* Loading the data set
* Computing data description in rows and column descriptions The data has 4700 rows and 46 columns
* Displaying the head and tail of the dataset
* Checking for null values in our data: There are a total of 5,587 missing values and no null values in the dataset.
   * But Notes(100%) and attendance(29%) are key contributors to these numbers.
* Checking for unique values in our dataset 
* Checking for duplicates in our dataset: We have no duplicated data points in the dataset.

## Exploratory  Data Analysis
### Univariate Analysis

**The total number of wins,draws and losses**
This is a visualisation of the total number of wins, losses and draws for the whole season
![download](https://github.com/WatiriVivian/Capstone_project_phase5_Group2/assets/118829983/518b124c-eccd-4ade-afe6-b7d72458a400)

**The Average attendance over all the seasons**
The graph below shows the game attendance over the last 4 seasons.
There was a drastic drop in the  attendance of games during the year 2020 and 2021 this was during the covid19 period. 
After that period there was a drastic pick in the number of fans attending the games since the covid19 restrictions were lifted.

![download](https://github.com/WatiriVivian/Capstone_project_phase5_Group2/assets/118829983/b272ea42-311a-4917-9705-e9a0e0af93bb)

### Bivariate Analysis

**Performance on each formation**
This allows us to view how teams perform while playing on specific formations:
We now know that teams perform the best on the 4-3-3
Despite the fact that teams had equal wins on the 4-2-3-1 formation they also have the highest number of losses on this formation
Also the 3-5-2 formation seems to have a lot more losses than wins so I would dub this the worst formation.

![download](https://github.com/WatiriVivian/Capstone_project_phase5_Group2/assets/118829983/5b416f0e-50e0-47d7-abbd-988382e37220)


**Wins at Home as compared to Away games.**
Team performance on away games as compared to home games          
Below is a comparison of the number of wins in away games as compared to the number of wins at home.
Apart from Leeds United and Luton Town  all the other teams seem to perform better at home games than away.
Again  Manchester city seems to have the most number of wins and even in this case most are home games.

![download](https://github.com/WatiriVivian/Capstone_project_phase5_Group2/assets/118829983/61a3d74c-7ee0-4a1d-881d-1bca126b50d8)

Performance compared to the attendance
Below is a comparison between the number of wins losses and draws and the average attendance per team
Manchester United games seems to have had the highest attendance average


![download](https://github.com/WatiriVivian/Capstone_project_phase5_Group2/assets/118829983/49b7603b-5835-4807-8e57-16f2fbc69fe6)

Modelling
To prepare our data for modelling we handled:
**Multicollinearity** by dropping highly correlated variables
**Label encoding variables** as we intend to use tree models boosting models which work well with label-encoded categorical variables
**Scaled numeric columns**, applied Principal Component Analysis (PCA) to reduce dimensionality, and used SMOTE to address class imbalance in the target variable.
Our best performing models

#### Logistic Regression
The logistic regression model is the foundation for predicting binary outcomes, specifically whether a match results in a win or not. It serves as a baseline, preparing the path for more advanced algorithms like tree models, boosting models, and neural networks capable of handling complex patterns and relationships.

The model's performance on the test data was as follows:

• Accuracy: 57.91%
• Precision: 58.17%
• Recall: 57.91%
• F1-score: 57.57% 

> * The Logistic Regression model achieved an accuracy of 57.91% and a precision of 58.17% on the test data.
> * Demonstrating its capability for reasonably accurate predictions and a decent true positive rate.
> *  However, it showed a lower recall of 57.91%, indicating that it missed some positive instances.
> *  The F1-score, at 57.57%, represents a fair balance between precision and recall, summarizing the model's overall performance on the test dataset.

 #### Neural Network
The Neural Networks model, implemented using MLPClassifier, represents a sophisticated approach to match outcome prediction. It was implemented as it is capable of capturing complex patterns in the data.

> * The Neural Networks model achieved a test data accuracy of 69.18% and a precision of 71.29%, indicating its capability for accurate predictions and a high true positive rate. 
> * However, it had a lower recall of 69.18%, indicating that it missed many positive instances.
> * The F1-score of 68.40 represents a reasonable balance between precision and recall, reflecting the model's performance on the test data

4. Gradient Boost
The Gradient Boosting Model, a robust ensemble learning technique, was utilized to predict match outcomes. This model builds multiple decision trees sequentially, where each tree corrects the errors of the previous one, leading to enhanced accuracy and predictive power.

The Gradient Boosting model performance on the test data was as follows:

Accuracy: 71.64%
Precision: 73.33%
Recall: 71.64%
F1-score: 71.11%
ROC AUC Score: 83.94%

The Gradient Boosting model exhibited strong predictive capabilities, outperforming most models and demonstrating the ability to capture complex relationships within the data. 

It maintained a high accuracy of 71.64% and precision of 73.33%, suggesting accurate predictions and a high proportion of true positives. 

The recall score of 71.64% indicates that the model successfully captured around 71.64% of the actual positive instances.

The F1-score of 0.7111 represents a good balance between precision.

##### Additional models we attempted
1. **Random Forest**
The Random Forest model, a powerful ensemble learning technique, was employed to predict match outcomes. Unlike logistic regression, Random Forest can capture non-linear relationships and interactions within the data due to its ability to work with multiple decision trees.
The model's performance on the test data was as follows:

Accuracy: 67.18%
Precision: 69.27%
Recall: 67.18%
F1-score: 66.27%
ROC AUC Score: 75.46%

The Random Forest model demonstrated strong predictive capabilities, outperforming the logistic regression baseline by its ability to capture non-linear relationships resulting in robust predictions. 

2. *&XGBOOST**
XGBoost, an efficient and scalable gradient boosting framework, was employed to predict match outcomes. This model, renowned for its speed and performance, iteratively builds multiple decision trees, allowing it to capture complex relationships in the data.
The model's performance on the test data was as follows:

Accuracy: 67.73%
Precision: 69.83%
Recall: 67.73%
F1-score: 66.85%
ROC AUC Score: 80.49%

The XGBoost model showcased impressive predictive power, surpassing the logistic regression baseline.
Model Evaluation
The Area Under the ROC Curve (AUC-ROC) was used to quantify the overall performance of the model. 




## Model selection 
The Gradient Boosting model outperforms others because it achieves a high level of accuracy, precision, and recall and a good balance between precision and recall (as indicated by the F1-score). 

Moreover, its **AUC score of 0.839** shows that it excels in distinguishing between different match outcomes, making it the best choice for predicting football match outcomes in your context.
Given that it had the highest precision, of 0.73 indicating a relatively low rate of false positive predictions. We chose the gradient boost model


![download](https://github.com/WatiriVivian/Capstone_project_phase5_Group2/assets/118829983/ac26dfbf-7ce0-4bd7-a327-a722bc01539c)

## Model Deployment



























Model Evaluation
The Area Under the ROC Curve (AUC-ROC) was used to quantify the overall performance of the model. 




Model selection 
The Gradient Boosting model outperforms others because it achieves a high level of accuracy, precision, and recall, as well as a good balance between precision and recall (as indicated by the F1-score). Moreover, its AUC score of 0.839 shows that it excels in distinguishing between different match outcomes, making it the best choice for predicting football match outcomes in your context.
Given that it had the highest precision,of 0.73 indicating a relatively low rate of false positive predictions. We choose the gradient boost model







