import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier as knn_cl
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score
plt.style.use('ggplot')
sns.set_style('darkgrid')

"""
In this project, we analyze the Pima Indians dataset
and try to predict whether or not a person has diabetes
based on their characteristics such as BMI, BloodPressure, Glucose level,
and so on. I will compare between KNN and Multiple Linear Regression
"""

pi_df = pd.read_csv("pima-indians-diabetes.csv") # load the data into a dataframe

# print(pi_df.head())

pi_df.columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
		"BMI", "DPF", "Age", "Class"] # assign column names

print(pi_df.head()) # print the first 5 rows

print("\n\n\n")

print(pi_df.describe()) # print the statistics of the data frame

print("\n\n\n")

y = pi_df["Class"] # let's define our target variable

x = pi_df[["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
		"BMI", "DPF", "Age"]] # define the features

print(y.head())
print("\n\n\n")
print(x.head())
print("\n\n\n")

sns.set(style='ticks')
plt.figure(figsize=(20,5))
sns.pairplot(pi_df, hue='Class')
fig1 = plt.gcf()
#plt.show()
plt.draw()
fig1.savefig("Pima_pairplot.png")


"""
Next, we check for zero values in our data
"""

print("Count of BloodPressure values equal to 0: ", pi_df[pi_df.BloodPressure == 0].shape[0])
print("Count of BMI values equal to 0: ", pi_df[pi_df.BMI == 0].shape[0])
print("Count of SkinThickness values equal to 0: ", pi_df[pi_df.SkinThickness == 0].shape[0])
print("Count of Insulin values equal to 0: ", pi_df[pi_df.Insulin == 0].shape[0])
print("Count of Glucose values equal to 0: ", pi_df[pi_df.Glucose == 0].shape[0])

"""
Since we have a large number of 0 values 
in SkinThickness and Insulin, we will replace the Insulin 0 values
with the mean of the data available. Same goes for skin thickness

We will remove the cases where value is 0 for BMI, Blood Pressure
and Glucose
"""

insulin_non_zero_mean = pi_df[pi_df.Insulin != 0].mean()
# First you can find the nonzero mean

pi_df.loc[pi_df.Insulin == 0, "Insulin"] = insulin_non_zero_mean
#replace

sk_th_nzmean = pi_df[pi_df.SkinThickness != 0].mean()
pi_df.loc[pi_df.SkinThickness == 0, "Insulin"] = sk_th_nzmean

"""
We now visualize the effect of the different features on the outcome
by using matplotlib
"""

# 1: BMI
plt.figure(figsize=(25,5))
BMI_pivot = pi_df.groupby('BMI').Class.mean().reset_index()
"""
we create a pivot table where we group BMI values against the Class
with missing values being excluded. reset_index creates a data frame 
with the index being reset to the original 0, 1, 2 ... N notation
"""
sns.barplot(BMI_pivot.BMI, BMI_pivot.Class) # create a barplot using seaborn
# alternatively, we could use BMI_pivot.plot(kind="bar", etc.)
plt.title('Percent chance (%) of having diabetes based on BMI')
fig2 = plt.gcf()
#plt.show()
plt.draw()
fig2.savefig("'BMI vs Class'.png")

#2: Glucose

plt.figure(figsize=(25,5))
glucose_pivot = pi_df.groupby('Glucose').Class.mean().reset_index()
sns.barplot(glucose_pivot.Glucose, glucose_pivot.Class)
plt.title('Percent chance (%) of having diabetes based on Glucose')
fig3 = plt.gcf()
#plt.show()
plt.draw()
fig3.savefig("'Glucose vs Class'.png")

#3: Blood Pressure

plt.figure(figsize=(25,5))
Blood_P_pivot = pi_df.groupby('BloodPressure').Class.mean().reset_index()
sns.barplot(Blood_P_pivot.BloodPressure, Blood_P_pivot.Class)
plt.title('Percent Chance (%) of having diabetes based on Blood Pressure')
fig4 = plt.gcf()
#plt.show()
plt.draw()
fig4.savefig("'Blood Pressure vs Class'.png")

#4: Insulin

plt.figure(figsize=(25,5))
insulin_pivot = pi_df.groupby('Insulin').Class.mean().reset_index()
sns.barplot(insulin_pivot.Insulin, insulin_pivot.Class)
plt.title('Percent chance (%) of having diabetes based on Insulin reading')
fig5 = plt.gcf()
#plt.show()
plt.draw()
fig5.savefig("'Insulin vs Class'.png")


"""
Now we split our data, train the models, and compare the accuracies
and the confusion matrices
"""

pi_df_new = pi_df[(pi_df.BloodPressure != 0) & (pi_df.BMI != 0) & (pi_df.Glucose!=0)] # create a new dataframe with only non-zero values for
pi_df_new.dropna(axis=0, inplace=True)
train_set, test_set = train_test_split(pi_df_new, test_size=0.25)

print(pi_df_new.shape)
print(train_set.shape)
print(test_set.shape)

print("\n\n\n")


features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness'\
            , 'BMI', 'Age', 'Insulin', 'DPF']
target = 'Class'

classifier_list = [knn_cl(), LinearRegression()]
classifier_names = ['K Nearest Neighbors', 'Linear Regression']

"""
We compare cross_validation accuracies of our classifiers with cv folds = 5
"""

for clf, clf_name in zip(classifier_list, classifier_names):
    cross_v_scores = cross_val_score(clf, train_set[features], train_set[target], cv=5)

    print(clf_name, ' average rounded accuracy: ', round(cross_v_scores.mean()*100, 3), '% std: ', round(cross_v_scores.var()*100, 3), '%')


knn_cl_finmod = knn_cl(n_neighbors=5).fit(train_set[features], train_set[target])

lin_reg_finmod = LinearRegression().fit(train_set[features], train_set[target])

y_pred_knn = knn_cl_finmod.predict(test_set[features])
y_pred_linreg = lin_reg_finmod.predict(test_set[features])

print("KNN Accuracy Score is: ", accuracy_score(test_set[target], y_pred_knn)*100, '%')

print("Linear Regression R^2 Score is: ", r2_score(test_set[target], y_pred_linreg))


"""
We can also try and find out the optimal value of K for the knn
"""

Ks=11
mean_acc=np.zeros((Ks-1))


#train and predict
for n in range(1,Ks):
    neighb=knn_cl(n_neighbors=n).fit(train_set[features],train_set[target])
    y_pred=neighb.predict(test_set[features])
    mean_acc[n-1]=accuracy_score(test_set[target],y_pred)


print(mean_acc)

print( "The best accuracy was with", mean_acc.max(), "with k = ", mean_acc.argmax()+1) 


"""
We print and save the confusion matrix for KNN

"""

# 1: KNN
plt.figure(figsize=(20,20))
plt.title("KNN Confusion Matrix")
sns.heatmap(confusion_matrix(test_set[target], y_pred_knn), annot=True, cmap="OrRd")
plt.xlabel('Predicted Classes')
plt.ylabel('True Classes')
fig7 = plt.gcf()
# plt.show()
plt.draw()
fig7.savefig("'KNN Confusion Matrix'.png")

# 2: Linear Regression
plt.figure(figsize=(20,20))
plt.title("Linear Regression Confusion Matrix")
cutoff = 0.5
y_pred_classes = np.zeros_like(y_pred_linreg)
y_pred_classes[y_pred_linreg>cutoff] = 1
sns.heatmap(confusion_matrix(test_set[target], y_pred_classes), annot=True, cmap="PuBu")
plt.xlabel("Predicted Classes")
plt.ylabel("True Classes")
fig8 = plt.gcf()
# plt.show()
plt.draw()
fig8.savefig("'Linear Regression Confusion Matrix'.png")


