# Predicting-Employee-Attrition-for-Augmenting-Institutional-Yield

Employee Attrition    |  
:-------------------------:|
![](Plots/employee_attrition.png) |

Employee Attrition can be defined as the natural process in which the employees of an organization or an institution leave the workforce and are not immediately replaced. The reason for attrition can range from personal reasons such as low salaries to hostile work environments. Employee attrition can be categorized into two categories, viz Voluntary Attrition and Involuntary Attrition. Voluntary attrition implies that the employee leaves an organization due to personal reasons. On the other hand, involuntary attrition occurs when an employee is removed from the organization due to low productivity or other reasons. Loss of employees via attrition has severe impacts on the yield of an organization. Finding eligible candidates to replace the ones that have left is a daunting task. This not only leads to higher costs but also induces a relatively inexperienced workforce in the organization. Continuous employee loss also disrupts the work chain and leads to delayed deadlines and lower customer satisfaction. Higher employee attrition diminishes the brand value of an organization.

---
## Outline

1. [Introduction](#introduction)
2. [Literature Survey](#literature-survey)
3. [Dataset](#dataset)
   * [Exploratory Data Analysis](#exploratory-data-analysis)
   * [Data Preparation and Preprocessing](#data-preparation-and-preprocessing)
4. [Methodology](#methodology)
   * [Classification Techniques](#classification-techniques)
   * [Evaluation](#evaluation)
5. [Results and Analysis](#results-and-analysis)
6. [Conclusions](#conclusions)
7. [References](#references)

---

## Project Description
### Introduction
In this project, the team strives to use machine learning principles to predict employee attrition, provide managerial insights to prevent attrition, and finally rule out and present the factors that lead to attrition. The project employs the use of models like Logistic Regression, Naive Bayes Classifier, Decision Trees, Random Forests, SVM and Multi Layer Perceptron on an IBM Watson generated synthetic data set.

### Literature Survey
The paper [1] starts off with describing what is employee attrition, and why it is a major issue faced by institutions across the globe. The paper aimed at predicting Voluntary Employee Attrition within a company using a K-Nearest Neighbours Algorithm, and compare its performance with other models, including Naive Bayes, Logistic Regression and NLP. The authors performed data preprocessing by converting categorical feature values into numerical ones, like converting salary values, that were either ”low”, ”medium” and ”high” to 0,1 and 2. A 70-30 train-test split was created on the data set, and the various model”s performances were evaluated using metrics like Area-Under-Curve, Accuracy and F1 Score. The results of this research showed the superiority of the KNN classifier in terms of accuracy and predictive effectiveness, by means of the ROC curve.  

The second paper [2] talked about how classification algorithms often perform unreliably on data sets with large sizes. These data-sets are also often prone to class imbalances, redundant features or noise. The paper applied dimensionality reduction by PCA on the Lung-Cancer dataset, which was followed by a SMOTE re-sampling to balance the different class distributions. This was followed by applying a Naive-Bayes Classifier on the modified data set, the performance of which was evaluated across four metrics: Overall accuracy, False Positive Rate, Precision and Recall.The results obtained showed that the least misclassifications occurred when PCA was applied followed by applying SMOTE re-sampling twice. Applying SMOTE twice balanced the distributions of the two minority classes, thus giving the best results.

### Dataset
The data set is a fictional data set created by IBM data scientists. It has 1470 instances and 34 features (27 numerical and 7 categorical) describing each employee. The target variable - "Attrition" is imbalanced. We have 83% of employees who have not left the company and 17% who have left the company. If one variable is highly correlated to another variable, it will lead to skewed or misleading results.  

Correlation Matrix   |  
:-------------------------:|
![](Plots/Confusion-Matrix.png) |

#### Exploratory Data Analysis

Attrition Rates   |  
:-------------------------:|
![](Plots/attrition.png) |

As seen from above plot, there is a severe class imbalance in the attrition class and the value of 'No' or 0 far outweigh the 'Yes' or 1.  

Age vs Attrition Rates   |  
:-------------------------:|
![](Plots/attrition_age.png) |

Younger people tend have to higher attrition rates.  

Department vs Attrition Rates   |  
:-------------------------:|
![](Plots/attrition_department.png) |

The Sales Department has the highest attrition with human resources being slightly less than it.  


Job Role vs Attrition Rates   |  
:-------------------------:|
![](Plots/attrition_job_role.png) |

The job role with the least attrition is of a ResearchDirector and the one with maximum attrition is of a Sales Representative.  

Gender vs Attrition Rates   |  
:-------------------------:|
![](Plots/attrition_gender.png) |

The Male employees tend to have a higher attrition rate as compared to females.

Marital Status vs Attrition Rates   |  
:-------------------------:|
![](Plots/attrition_marital_status.png) |

Single employees tend to have higher attrition. 

Overtime vs Attrition Rates   |  
:-------------------------:|
![](Plots/attrition_overtime.png) |

People who overtime have a higher chances of leaving the organization.     

Holistic Satisfaction vs Attrition Rates   |  
:-------------------------:|
![](Plots/attrition_holistic.png) |

Employees with less Holistic Satisfaction tend to leave the organization.   

Monthly Income vs Attrition Rates   |  
:-------------------------:|
![](Plots/attrition_income.png) |

People with lower monthly income have a higher attrition rate.

### References
1. [Rahul Yedida, Rahul Reddy, Rakshit Vahi, Rahul Jana, Abhilash GV, Deepti Kulkarni. "Employee Attrition Prediction". 02 November 2018.](https://arxiv.org/abs/1806.10480)
2. [Mehdi Naseriparsa, Mohammad Mansour Riahi Kashani "Combination of PCA with SMOTE Resampling to Boost the Prediction Rate in Lung Cancer Dataset".](https://arxiv.org/abs/1403.1949)
