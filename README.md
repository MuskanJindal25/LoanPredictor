# LoanPredictor
ML Project to predict loan status

In this project, I have worked on a labelled dataset for supervised learning to find out whether an individual should be given a loan or not.

The features of my dataset are: 
- Gender 
- Married or not 
- No. of dependencies
- Education qualification
- Applicant income
- Co-applicant income
- Loan amount
- Loan amount term
- Credit history
- Area of property
- Loan status

Various classifiers used to make this prediction are:
- Logistic Regression
- Decision trees
- Random Forest
- Support Vector Machine (SVM)
- Gaussian Naive Bayes
- k- Nearest Neighbors
- Gradient Boosting

### Objective:- Compare how different models work on given dataset

Major Steps involved :
- Data Cleaning
- Data Normalization
- Data Spliting (split size= 0.2)
- Model Training 
- Testing 

Data cleaning was performed as there were multiple rows having null entries for some features. So, the numerical values were replaced with mean while the categorical values were replaced with mode for the respective columns. 
Further I normalised the feature- loan amount term values in order to convert it in a certain range.

The above mentioned models were then trained on training dataset and used for predicting loan status on testing dataset. Accuracy was calculated on the basis of how many predictions were correct using confusion matrix.

On average highest accuracies are obtained from 
1. Logistic Regression 
2. Gaussian Naive Bayes algorithms
