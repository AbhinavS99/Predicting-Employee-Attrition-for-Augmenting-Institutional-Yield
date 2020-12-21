# Predicting-Employee-Attrition-for-Augmenting-Institutional-Yield

<h2>Introduction</h2>
Employee Attrition can be defined as the natural process in which the employees of an organization or an institution leave the workforce and are not immediately replaced. The reason for attrition can range from personal reasons such as low salaries to hostile work environments. Employee attrition can be categorized into two categories, viz Voluntary Attrition and Involuntary Attrition. In this project, we used machine learning principles to predict employee attrition, provide managerial insights to prevent attrition, and finally rule out and present the factors that lead to attrition. The project employs the use of models like Logistic Regression, Naive Bayes Classifier, Decision Trees, Random Forests, SVM and Multi Layer Perceptron on an IBM Watson generated synthetic data set.
</br>
<h2>Dataset</h2>
The dataset is a fictionsl data-set created by IBM data scientists and can be found on kaggle: https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset
<br>
<h2>Objective</h2>
Using machine learning algorithms to predict employee attrition in Python.
<br>
<h2>Approach and Methodology</h2>
Visualization of the dataset is done to clearly understand all the attributes or features used for predicting the attrition of employees. The data-set is highly imbalanced, therefore, for all classification algorithms three separate instances of data-set are prepared viz baseline(without any change), up-sampling the minority class using SMOTE and down-sampling the majority class using sklearn resample. Then the classification methods are applied on all the three instances. The classification techniques that have been used within the domain of the project are Logistic Regression, Gaussian Naive Bayes, Decision Tree Classifier, Random Forest Classifier, Perceptron, Multi-Layer Perceptron and SVM/SVC. The models are evaluated using the evaluation metrics Precision and Recall. Also, ROC and PR curves are used to visualize and evaluate all the models. Random forest was later on used to extract the most importance features to predict employee attrition.
<br>
 
 <h2>Conclusion</h2>
We achieved the best precision of 91.75 using Random Forest Classifier in baseline setting. However, Logistic Regression, SVC, XGBoost, and MLP also had precision ranging from 80 to 90.
The important features extracted after application of random forest were monthly income, age, overtime and holistic satisfaction.
