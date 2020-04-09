# LoanPredictor
ML Project to predict loan status

In this project, I have worked on a dataset to find out whether an individual should be given loan or not. My model uses supervised learning to make this prediction. 

Various classifiers used to make this prediction are:
- Logistic Regression
- Decision trees
- Random Forest
- Support Vector Machine (SVM)
- Gaussian Naive Bayes
- k- Nearest Neighbors
- Gradient Boosting

On average highest accuracies are obtained from 
1. Logistic Regression 
2. Gaussian Naive Bayes algorithms

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

To carry out this project the approach followed is, I read the data from a csv file and observed that there were multiple rows having null entries, so to overcome that I used dataframe.fillna() technique. The numerical values were replaced with mean while the categorical values were replaced with mode for the respective columns. Further I have normalised the loan amount term values in order to …-.-- .  Used pandas.get_dummies() for converting to categorical data into indicator variables. Now the most important step was to convert our dataset into training and testing datasets. Every model used is trained on this training dataset and it’s accuracy is obtained using testing dataset. For every model used we first import it from sklearn and then fit our model over training data and then use it for predicting our loan status. The accuracy for each is calculated using a confusion matrix which tells us how many true positives, false positives, false negatives and true negatives are present in our predicted values. Finally we convert our dataframe for all our predictions into a csv file containing the loan id and corresponding loan status for each model used.
