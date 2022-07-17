##########################################################
# Import libraries and dataset
##########################################################

##Remove all objects
rm(list=ls())

##Import Libraries
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(rpart)) install.packages("rpart", repos = "http://cran.us.r-project.org")
if(!require(MASS)) install.packages("MASS", repos = "http://cran.us.r-project.org")
if(!require(pROC)) install.packages("pROC", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(scales)) install.packages("scales", repos = "http://cran.us.r-project.org")
if(!require(rpart.plot)) install.packages("rpart.plot", repos = "http://cran.us.r-project.org")

library(tidyverse) # for data manipulation
library(caret)  # for training machine learning models
library(data.table) #  working with tabular data
library(rpart)  # for decision tree model
library(MASS)  # for generative model: qda
library(pROC)  # for possible ROC curve plotting
library(randomForest)  # for randomForest model
library(scales)    # to compute percentages in ggplot
library(rpart.plot)  # for decision tree plotting 


library(readr) # to read csv file
##Obtain url for data
urlfile <- 'https://raw.githubusercontent.com/EmmaAntiri/Own_Project/main/data.csv'
data <- read_csv(urlfile) # read csv from the urlfile 


###Load dataset (Alternatively you can download and set directory)
#data <- read_csv("data.csv")


##Ascertain the structure of dataset
str(data)
#there are 60000 observations and 9 variables.
# there are 2 character variables (gender and region) & 7 numeric (double) variables. 

##Checking the head to ascertain it loaded well
head(data)


#Create dataframe with variable and description
variable <- colnames(data)
description <- c("Level of glucose (sugar) in blood measured in mmol/L", 
                 "Average bilirubin concentration measured in micromol/L",
                 "Creatinine Concentration in blood measured in micromol/L",
                 "Average Blood Urea Nitrogen Concentration in serum measured in millimol/L",
                 "Age of Person measured in years", 
                 "Average platelets count in blood measured in microliter",
                 "Gender of the Person (Male/Female)", 
                 "Region of Person (North/South)",
                 "Disease Status (1 meaning Diagnosed/0 meaning Not Diagnosed)")

variable_description <- data.frame(variable, description) #create dataframe with variable name and description
variable_description %>% knitr::kable()  ##show the variable description


#Check for presence of missing data
sapply(data, function(x) sum(is.na(x)))
##There are no missing values. 


##Convert outcome variable column to factor for easy computation
data$disease_status <- factor(data$disease_status, 
                              levels = c(0, 1), 
                              labels = c('Not_Diagnosed', 'Diagnosed')) 


##########################################################
# Create test and train set
##########################################################

##Split data into train/test split 
# Set seed for reproducible results
set.seed(105, sample.kind = "Rounding")    # if using R 3.6 or later
test_index <- createDataPartition(data$disease_status, times = 1, p = 0.2, list = FALSE) # Random Splitting into 80% train/20% test
train <- data[-test_index,] # Obtain train set using the index created. 
test <- data[test_index,] # Obtain test set using the index created.

## The algorithm would be train exclusively on the training set and the test set would be used for evaluation.
## Thus, the train set contains a large portion of the dataset (80%) as compared to the test set (20%)
## We need as much dataset to train (80%) and we need a good portion to evaluate on (20%); so to obtain stable estimates. 


#Exploratory Data Analysis (As with the MovieLens Project it would be done on the train set)
#Additionally, the exploratory data analysis is to understand the distribution of variables in relation to the outcome variable. 
#the train set was obtained through random splitting of 80% and would likely exhibit the distribution of the variable. 

# Summary Statistics of numeric columns (using lappy and sapply taught in Course 1)
summary_data <- sapply(train[unlist(lapply(train, is.numeric))], 
       quantile, na.rm=TRUE)  #lapply function used to select only continuous/numeric columns. 
#sapply is used to apply quantile on the selected continuous/numeric columns. 
summary_data <- as.data.frame(summary_data) # Convert the matrix created to dataframe

mean <- sapply(train[unlist(lapply(train, is.numeric))], 
               mean, na.rm=TRUE) # Same procedure to find the mean values of continuous columns
mean <- as.data.frame(mean) #Convert the matrix to dataframe. 
mean <- t(mean) #Transpose the column dataframe to a row dataframe

summary_data <- rbind(summary_data, mean) #combine quantiles to mean statistics
summary_data %>% knitr::kable() #showcase the metrics. 


##Summary Statistics for Diagnosed Group (Using Subsetting From Course 1)
# Summary Statistics of numeric columns (using lappy and sapply taught in Course 1)
summary_diagnosed <- sapply(train[train$disease_status == "Diagnosed",]
                            [unlist(lapply(train[train$disease_status == "Diagnosed",],is.numeric))], 
                       quantile, na.rm=TRUE)  #lapply function used to select only continuous/numeric columns. 
#sapply is used to apply quantile on all the continuous/numeric columns. 
summary_diagnosed <- as.data.frame(summary_diagnosed) # Convert the matrix created to dataframe

mean <- sapply(train[train$disease_status == "Diagnosed",]
                         [unlist(lapply(train[train$disease_status == "Diagnosed",], is.numeric))], 
               mean, na.rm=TRUE) # Same procedure to find the mean values of continuous columns
mean <- as.data.frame(mean) #Convert the matrix to dataframe. 
mean <- t(mean) #Transpose the column dataframe to a row dataframe

summary_diagnosed <- rbind(summary_diagnosed, mean) #combine quantiles to mean statistics
summary_diagnosed %>% knitr::kable() #showcase the metrics.



##Summary Statistics for Not Diagnosed Group (Using Subsetting From Course 1)
# Summary Statistics of numeric columns (using lappy and sapply taught in Course 1)
summary_notdiagnosed <- sapply(train[train$disease_status == "Not_Diagnosed",]
                            [unlist(lapply(train[train$disease_status == "Not_Diagnosed",],is.numeric))], 
                            quantile, na.rm=TRUE)  #lapply function used to select only continuous columns. 
#sapply is used to apply quantile on all the continuous columns created. 
summary_notdiagnosed <- as.data.frame(summary_notdiagnosed) # Convert the matrix created to dataframe

mean <- sapply(train[train$disease_status == "Not_Diagnosed",]
                         [unlist(lapply(train[train$disease_status == "Not_Diagnosed",], is.numeric))], 
                         mean, na.rm=TRUE) # Same procedure to find the mean values of continuous columns
mean <- as.data.frame(mean) #Convert the matrix to dataframe. 
mean <- t(mean) #Transpose the column dataframe to a row dataframe

summary_notdiagnosed <- rbind(summary_notdiagnosed, mean) #combine quantiles to mean statistics
summary_notdiagnosed %>% knitr::kable() #showcase the metrics.



##########################################################
# Exploratory Data Analysis 
##########################################################

##Outcome Variable 
##Disease Status 
##Plot bar plot on Outcome variable using ggplot
#the code allows for plotting in percentages. 
train %>% ggplot(aes(x = disease_status)) + 
  geom_bar(aes(y = (..count..)/sum(..count..))) +
  geom_text(aes(y=((..count..)/sum(..count..)), label = scales::percent((..count..)/sum(..count..))),stat = "count", vjust = -0.25) +
  scale_y_continuous(labels = percent) +
  theme_light() +
  labs(title = "Figure 1. Disease Status in Percentage", y = "Percent", x = "Disease Status") +
  theme(plot.title = element_text(face = "bold"))
##aesthetic set to disease_status(for plotting), geom_bar for barplot, y = (..count..)/sum(..count..) to calculate percentage (count/sum of count)
#geom_text to add percentage in plot; scale_y_continuous to set y to percentage, theme_light() to generate light/white background theme
#labs to set title, x and y labels 
#theme(plot.title = element_text(face = "bold")) to bold title text. 


#Gender 
#the code allows for plotting in percentages. 
train %>% ggplot(aes(x = gender)) + 
  geom_bar(aes(y = (..count..)/sum(..count..))) +
  geom_text(aes(y=((..count..)/sum(..count..)), label = scales::percent((..count..)/sum(..count..))),stat = "count", vjust = -0.25) +
  scale_y_continuous(labels = percent) +
  theme_light() +
  labs(title = "Figure 2. Gender in Percentage", y = "Percent", x = "Gender") +
  theme(plot.title = element_text(face = "bold"))


#Region
#the code allows for plotting in percentages. 
train %>% ggplot(aes(x = region)) + 
  geom_bar(aes(y = (..count..)/sum(..count..))) +
  geom_text(aes(y=((..count..)/sum(..count..)), label = scales::percent((..count..)/sum(..count..))),stat = "count", vjust = -0.25) +
  scale_y_continuous(labels = percent) +
  theme_light() +
  labs(title = "Figure 3. Region in Percentage", y = "Percent", x = "Region") +
  theme(plot.title = element_text(face = "bold"))


##Using Facet Wrap Function (Gender on Disease Status)
train %>% ggplot(aes(x = disease_status, fill = disease_status, y = gender)) + 
  geom_bar(aes(y = (..count..)/sum(..count..)), position = "dodge") +
  geom_text(aes(y=((..count..)/sum(..count..)), label = scales::percent((..count..)/sum(..count..))),stat = "count", vjust = -0.25) +
  scale_y_continuous(labels = percent) +
  theme_light() +
  labs(title = "Figure 4. Percentage of Disease Status by Gender", y = "Percent", x = "Gender") +
  facet_wrap(~gender) + 
  theme(plot.title = element_text(face = "bold"))
#facet_wrap to generate plots by group (gender)


##Using Facet Wrap Function (Region on Disease Status)
train %>% ggplot(aes(x = disease_status, fill = disease_status, y = region)) + 
  geom_bar(aes(y = (..count..)/sum(..count..)), position = "dodge") +
  geom_text(aes(y=((..count..)/sum(..count..)), label = scales::percent((..count..)/sum(..count..))),stat = "count", vjust = -0.25) +
  scale_y_continuous(labels = percent) +
  theme_light() +
  labs(title = "Figure 5. Percentage of Disease Status by Region", y = "Percent", x = "Region") +
  facet_wrap(~region) + 
  theme(plot.title = element_text(face = "bold"))
#facet_wrap to generate plots by group (Region)



##Distribution of Glucose Level
##Histogram 
# Plotting the Distribution of Glocuse; using ggplot 
train %>% 
  ggplot(aes(x = glucose_level)) +
  geom_histogram(bins = 50, color = "blue") + 
  theme_light() + 
  ggtitle("Figure 6. Distribution of Glucose Level") + 
  xlab("Glucose Level") +
  theme(plot.title = element_text(face = "bold"))
##geom_histogram to generate histogram

  
##Distribution of Glucose Level by Disease Status
# Using Density Plot 
train %>% 
  ggplot(aes(x = glucose_level, fill = disease_status)) +
  geom_density(alpha = 0.8, color = NA) + 
  scale_fill_manual(values = c("blue", "orange")) +
  theme_light() + 
  labs(title = "Figure 7. Density plot of Glucose Level on Disease Status", y = "Density", x = "Glucose Level") +
  theme(plot.title = element_text(face = "bold"))
#geom_density to generate density plot


##Distribution of Glucose Level by Disease Status 
# Using Box plot 
train %>% 
  ggplot(aes(x = disease_status, y = glucose_level, fill = disease_status)) + 
  geom_boxplot() + 
  theme_light() + 
  labs(title = "Figure 8. Box plot of Glucose Level on Disease Status", y = "Glucose Level", x = "Disease Status") +
  theme(plot.title = element_text(face = "bold"))
# geom_boxplot to generate box plot


#Bilirubin 
##Distribution of Bilirubin
##Histogram 
# Plotting the Distribution of Bilirubin; using ggplot 
train %>% 
  ggplot(aes(x = bilirubin)) +
  geom_histogram(bins = 50, color = "blue") + 
  theme_light() + 
  ggtitle("Figure 9. Distribution of Bilirubin") + 
  xlab("Bilirubin") +
  theme(plot.title = element_text(face = "bold"))


##Distribution of Bilirubin by Disease Status
# Using Density Plot 
train %>% 
  ggplot(aes(x = bilirubin, fill = disease_status)) +
  geom_density(alpha = 0.8, color = NA) + 
  scale_fill_manual(values = c("blue", "orange")) +
  theme_light() + 
  labs(title = "Figure 10. Density plot of Bilirubin on Disease Status", y = "Density", x = "Bilirubin") +
  theme(plot.title = element_text(face = "bold"))


##Distribution of Bilirubin by Disease Status 
# Using Box plot 
train %>% 
  ggplot(aes(x = disease_status, y = bilirubin, fill = disease_status)) + 
  geom_boxplot() + 
  theme_light() + 
  labs(title = "Figure 11. Box plot of bilirubin on Disease Status", y = "Bilirubin", x = "Disease Status") +
  theme(plot.title = element_text(face = "bold"))



#Creatinine Level
##Distribution of Creatinine Level
##Histogram 
# Plotting the Distribution of Creatinine Level; using ggplot 
train %>% 
  ggplot(aes(x = creatinine_level)) +
  geom_histogram(bins = 50, color = "blue") + 
  theme_light() + 
  ggtitle("Figure 12. Distribution of Creatinine Level") + 
  xlab("Creatinine Level") +
  theme(plot.title = element_text(face = "bold"))


##Distribution of Creatinine Level by Disease Status
# Using Density Plot 
train %>% 
  ggplot(aes(x = creatinine_level, fill = disease_status)) +
  geom_density(alpha = 0.8, color = NA) + 
  scale_fill_manual(values = c("blue", "orange")) +
  theme_light() + 
  labs(title = "Figure 13. Density plot of Creatinine Level on Disease Status", y = "Density", x = "Creatinine Level") +
  theme(plot.title = element_text(face = "bold"))


##Distribution of Creatinine Level by Disease Status 
# Using Box plot 
train %>% 
  ggplot(aes(x = disease_status, y = creatinine_level, fill = disease_status)) + 
  geom_boxplot() + 
  theme_light() + 
  labs(title = "Figure 14. Box plot of Creatinine Level on Disease Status", y = "Creatinine Level", x = "Disease Status") +
  theme(plot.title = element_text(face = "bold"))


#Blood Urea Nitrogen
##Distribution of Blood Urea Nitrogen
##Histogram 
# Plotting the Distribution of Blood Urea Nitrogen; using ggplot 
train %>% 
  ggplot(aes(x = blood_urea_nitrogen)) +
  geom_histogram(bins = 50, color = "blue") + 
  theme_light() + 
  ggtitle("Figure 15. Distribution of Blood Urea Nitrogen") + 
  xlab("Blood Urea Nitrogen") +
  theme(plot.title = element_text(face = "bold"))


##Distribution of Blood Urea Nitrogen by Disease Status
# Using Density Plot 
train %>% 
  ggplot(aes(x = blood_urea_nitrogen, fill = disease_status)) +
  geom_density(alpha = 0.8, color = NA) + 
  scale_fill_manual(values = c("blue", "orange")) +
  theme_light() + 
  labs(title = "Figure 16. Density plot of Blood Urea Nitrogen on Disease Status", y = "Density", x = "Blood Urea Nitrogen") +
  theme(plot.title = element_text(face = "bold"))


##Distribution of Blood Urea Nitrogen by Disease Status 
# Using Box plot 
train %>% 
  ggplot(aes(x = disease_status, y = blood_urea_nitrogen, fill = disease_status)) + 
  geom_boxplot() + 
  theme_light() + 
  labs(title = "Figure 17. Box plot of Blood Urea Nitrogen on Disease Status", y = "Blood Urea Nitrogen", x = "Disease Status") +
  theme(plot.title = element_text(face = "bold"))



#Age
##Distribution of Age
##Histogram 
# Plotting the Distribution of Age; using ggplot 
train %>% 
  ggplot(aes(x = age)) +
  geom_histogram(bins = 20, color = "blue") + 
  theme_light() + 
  ggtitle("Figure 18. Distribution of Age") + 
  xlab("Age") +
  theme(plot.title = element_text(face = "bold"))


##Distribution of Age by Disease Status
# Using Density Plot 
train %>% 
  ggplot(aes(x = age, fill = disease_status)) +
  geom_density(alpha = 0.8, color = NA) + 
  scale_fill_manual(values = c("blue", "orange")) +
  theme_light() + 
  labs(title = "Figure 19. Density plot of Age on Disease Status", y = "Density", x = "Age") +
  theme(plot.title = element_text(face = "bold"))


##Distribution of Age by Disease Status 
# Using Box plot 
train %>% 
  ggplot(aes(x = disease_status, y = age, fill = disease_status)) + 
  geom_boxplot() + 
  theme_light() + 
  labs(title = "Figure 20. Box plot of Age on Disease Status", y = "Age", x = "Disease Status") +
  theme(plot.title = element_text(face = "bold"))


#Platelets Counts
##Distribution of Platelets Counts
##Histogram 
# Plotting the Distribution of Platelets Counts; using ggplot 
train %>% 
  ggplot(aes(x = platelets_count)) +
  geom_histogram(bins = 50, color = "blue") + 
  theme_light() + 
  ggtitle("Figure 21. Distribution of Platelets Count") + 
  xlab("Platelets Counts") +
  theme(plot.title = element_text(face = "bold"))


##Distribution of Platelets Counts by Disease Status
# Using Density Plot 
train %>% 
  ggplot(aes(x = platelets_count, fill = disease_status)) +
  geom_density(alpha = 0.8, color = NA) + 
  scale_fill_manual(values = c("blue", "orange")) +
  theme_light() + 
  labs(title = "Figure 22. Density plot of Platelets Count on Disease Status", y = "Density", x = "Platelets Count") +
  theme(plot.title = element_text(face = "bold"))


##Distribution of Platelets Counts by Disease Status 
# Using Box plot 
train %>% 
  ggplot(aes(x = disease_status, y = platelets_count, fill = disease_status)) + 
  geom_boxplot() + 
  theme_light() + 
  labs(title = "Figure 23. Box plot of Platelets Count on Disease Status", y = "Platelets Count", x = "Disease Status") +
  theme(plot.title = element_text(face = "bold"))


################################################
#Pre-Processing 
################################################

##Separate features from outcome variable and preprocess features in both train and test split
train_y <- train$disease_status # outcome variable in train split 
train_x <- train[,colnames(train) != "disease_status"]  # features in train split 
test_y <- test$disease_status # outcome variable in test split 
test_x <- test[, colnames(test) != "disease_status"] # features in test split 

##Preprocessing for machine learning 
##MinMax scaling of continuous columns 
#Train Set
train_process <- preProcess(as.data.frame(train_x), method=c("range")) # Scale the continuous columns to be in range (0,1)
train_scaled <- predict(train_process, as.data.frame(train_x)) # Apply the scaling to the train data and store into object 'train_scaled'
#Test Set
test_process <- preProcess(as.data.frame(test_x), method=c("range")) #Scale the continuous columns to be in range (0,1)
test_scaled <- predict(test_process, as.data.frame(test_x)) # Apply the scaling to the test data and store into object 'test_scaled'


##One-hot Encoding: Converting string variables to dummy variables 
train_onehot <- dummyVars(" ~ .", data=train_scaled) #Convert string variables in train data to dummies
train_set <- data.frame(predict(train_onehot, newdata = train_scaled)) # Combine one-hot encoded variables and dummies into object 'train_set'
test_onehot <- dummyVars(" ~ .", data=test_scaled) # Convert string variables in test data to dummies 
test_set <- data.frame(predict(test_onehot, newdata = test_scaled)) # Combine one-hot encoded variables and dummies into object 'test_set'
##the binary variables Gender and Location were preprocessed. Remove one each of the other category

#Remove the binary dummy each.  
train_set$genderF <- NULL 
train_set$regionnorth <- NULL 
test_set <- test_set[,colnames(train_set)] 

##Remove redundant objects to save space
rm(test_index, test_onehot, test_process, test_scaled, train_onehot, train_process, train_scaled, test_x, train_x)


##Combine outcome and feature columns; I prefer it this way (though I could continue without combining)
train <- cbind(train_set, train_y) #Combine the train set
train$disease_status <- train$train_y  #Rename train_y to disease status
train$train_y <- NULL  #Remove column with name train_y
test <- cbind(test_set, test_y) # Combine the test split 
test$disease_status <- test$test_y  #Rename train_y to disease status
test$test_y <- NULL  # Remove column with name test_y


##Remove redundant objects to save space 
rm(test_y, train_y, test_set, train_set)

##Ascertain that the right columns are contained in train and test set 
print("Snapshot of the test dataset")
str(test)
print("Snapshot of the train dataset")
str(train)

#########################################
# Machine Learning Models 
#########################################
#Set default cross_validation for all models
control = trainControl(method = "cv", 
                       number = 5, 
                       classProbs = TRUE, 
                       summaryFunction = twoClassSummary, 
                       savePredictions = T)   
#5 fold cross-validation; twoClassSummary for binary classification; savePredictions = T to obtain predictions
#classProbs = TRUE to save probabilities for further analysis if need be. 

#Model 1: Guessing
set.seed(105, sample.kind = "Rounding") # set random seed for reproducibility for R 3.6 or later 
p <- mean(train$disease_status == "Diagnosed") #Find probability of guessing Diagnosed 
n <- length(test$disease_status) # Find the length of the test set 
y_hat <- sample(c("Not_Diagnosed", "Diagnosed"), 
                n, 
                replace = TRUE, 
                prob=c(1-p, p)) %>%
  factor(levels = levels(test$disease_status)) ##Generate prediction in the guessing model
cm_guessing <- confusionMatrix(y_hat, 
                               test$disease_status, 
                               positive = 'Diagnosed', 
                               mode = 'everything')  ##Generate confusion Matrix of the model
#The relevant metrics for this report are accuracy, balanced accuracy and F1-Score
# accuracy = cm_guessing$overall['Accuracy']
# balanced_accuracy = cm_guessing$byClass['Balanced Accuracy']
# F1_score = cm_guessing$byClass['F1']


# Model 2: Logistics Regression 
## Fit the logistics regression using the caret model 
getModelInfo("glm")$glm$parameters  # Ascertain whether there are parameters to be tuned. 
# there are no parameters to tune

set.seed(105, sample.kind = "Rounding") # set random seed for reproducibility for R 3.6 or later 
train_lr <- train(disease_status ~ ., 
                  data = train, 
                  method = "glm", 
                  family = "binomial", 
                  trControl = control) #Fit the model

##Variable Importance in Logistics Regression (train data)##Variable Importance in Logistics Regression (train data)
varImp(train_lr)
plot(varImp(train_lr), main = "Figure 24. Plot of Variable Importance of Logistics Model")
##Summary Results 
summary(train_lr)


##Prediction and confusion Matrix
lr_preds <- predict(train_lr, test) # Obtain prediction 
cm_lr <- confusionMatrix(lr_preds, 
                         test$disease_status, 
                         positive = 'Diagnosed', 
                         mode = 'everything')
#The relevant metrics for this report are accuracy, balanced accuracy and F1-Score
# accuracy = cm_lr$overall['Accuracy']
# balanced_accuracy = cm_lr$byClass['Balanced Accuracy']
# F1_score = cm_lr$byClass['F1']



##Model 3: KNN
getModelInfo("knn")$knn$parameters #Ascertain whether there are parameters to be tuned. 
#the parameter to be tuned is number of neighbours (k)

set.seed(105, sample.kind = "Rounding") # set random seed for reproducibility for R 3.6 or later 
train_knn <- train(disease_status ~ ., 
                   data = train, 
                   method = "knn",  
                   trControl = control, 
                   tuneGrid = data.frame(k = seq(9, 33, 2))) #Fit the model with tuning of k

train_knn$bestTune  # the best tuned knn

ggplot(train_knn, highlight = TRUE) +
  theme_light() +
  labs(title = "Figure 25. Tuning performance of KNN neighbours") +
  theme(plot.title = element_text(face = "bold"))



knn_preds <- predict(train_knn, test) # Obtain prediction 
cm_knn <- confusionMatrix(knn_preds, 
                          test$disease_status, 
                          positive = 'Diagnosed', 
                          mode = 'everything') #Generate confusion Matrix of the model
#The relevant metrics for this report are accuracy, balanced accuracy and F1-Score
# accuracy = cm_knn$overall['Accuracy']
# balanced_accuracy = cm_knn$byClass['Balanced Accuracy']
# F1_score = cm_knn$byClass['F1']



##Model 4: QDA (Generative Model)
getModelInfo("qda")$qda$parameters  #Ascertain whether there are parameters to be tuned. 
#there are no parameters to be tuned. 

set.seed(105, sample.kind = "Rounding") # set random seed for reproducibility for R 3.6 or later
train_qda <- train(disease_status ~ ., 
                     method = "qda", 
                     trControl = control, 
                     data = train) # fit the model 

#plot the model showing the best prediction
qda_preds <- predict(train_qda, test) # Obtain prediction 
cm_qda <- confusionMatrix(qda_preds, 
                          test$disease_status, 
                          positive = 'Diagnosed', 
                          mode = 'everything') #Generate confusion Matrix of the model
#The relevant metrics for this report are accuracy, balanced accuracy and F1-Score
# accuracy = cm_qda$overall['Accuracy']
# balanced_accuracy = cm_qda$byClass['Balanced Accuracy']
# F1_score = cm_qda$byClass['F1']

##Model 5: Decision Tree
getModelInfo("rpart")$rpart$parameters #Ascertain whether there are parameters to be tuned
# the parameter to be tuned is cp. 

set.seed(105, sample.kind = "Rounding") # set random seed for reproducibility for R 3.6 or later
train_dtree <- train(disease_status ~., 
                     data = train, 
                     method = 'rpart', 
                     tuneGrid = data.frame(cp = seq(0, 0.05, 0.002))) #Fit the decision tree by tuning cp
train_dtree$bestTune # the best tuned is cp = 0.002

##Variable Importance
varImp(train_dtree)
plot(varImp(train_dtree), main = "Figure 26. Plot of Variable Importance of Decision Tree Model")

#Plot the final decision tree
prp(train_dtree$finalModel, type = 1, extra = 1, split.font = 1, varlen = -10, 
    main = "Figure 27. Classification Tree for Disease Diagnostics")


dtree_preds <- predict(train_dtree, test) #Obtain prediction 
cm_dtree <- confusionMatrix(dtree_preds, 
                            test$disease_status, 
                            positive = 'Diagnosed', 
                            mode = 'everything') #Obtain confusion Matrix
#The relevant metrics for this report are accuracy, balanced accuracy and F1-Score
# accuracy = cm_dtree$overall['Accuracy']
# balanced_accuracy = cm_dtree$byClass['Balanced Accuracy']
# F1_score = cm_dtree$byClass['F1']



##Model 6: Random Forest 
getModelInfo("rf")$rf$parameters #Ascertain whether there are parameters to be tuned
# the parameter to be tuned is mtry.

# set.seed(105)
set.seed(105, sample.kind = "Rounding") # set random seed for reproducibility for R 3.6 or later
train_rnf <- train(disease_status ~ ., 
                   data = train,
                   method = "rf",
                   tuneGrid = data.frame(mtry = seq(1:5)),
                   trControl = control,
                   ntree = 500,
                   importance = TRUE) #Fit the model by tuning mtry parameter with 500 trees. 
##Set importance = TRUE to obtain variable importance

train_rnf$bestTune #Ascertain the best tuned parameters 

##Ascertain Variable Importance
plot(varImp(train_rnf), main = "Figure 28. Plot of Variable Importance of Random Forest Model")
varImp(train_rnf)


rnf_preds <- predict(train_rnf, test) ## Obtain prediction 
cm_rnf <- confusionMatrix(rnf_preds, 
                          test$disease_status, 
                          positive = 'Diagnosed', 
                          mode = 'everything') #Obtain confusion Matrix
#The relevant metrics for this report are accuracy, balanced accuracy and F1-Score
# accuracy = cm_rnf$overall['Accuracy']
# balanced_accuracy = cm_rnf$byClass['Balanced Accuracy']
# F1_score = cm_rnf$byClass['F1']


############################
# Results 
############################
##Create dataframe of Performance Metrics of the Models
Method <- c('Model 1: Baseline(Guessing)', 'Model 2: Logistics Regression', 'Model 3: KNN', 'Model 4: QDA', 
            'Model 5: Decision Tree', 'Model 6: Random Forest') #Create row names

Accuracy <- c(round(cm_guessing$overall['Accuracy'], 4), round(cm_lr$overall['Accuracy'], 4), 
              round(cm_knn$overall['Accuracy'], 4), round(cm_qda$overall['Accuracy'], 4), 
              round(cm_dtree$overall['Accuracy'], 4), round(cm_rnf$overall['Accuracy'], 4))

Balanced_Accuracy <- c(round(cm_guessing$byClass['Balanced Accuracy'], 4), round(cm_lr$byClass['Balanced Accuracy'], 4), 
                       round(cm_knn$byClass['Balanced Accuracy'], 4), round(cm_qda$byClass['Balanced Accuracy'], 4), 
                       round(cm_dtree$byClass['Balanced Accuracy'], 4), round(cm_rnf$byClass['Balanced Accuracy'], 4))

F1_Score <- c(round(cm_guessing$byClass['F1'], 4), round(cm_lr$byClass['F1'], 4), 
                       round(cm_knn$byClass['F1'], 4), round(cm_qda$byClass['F1'], 4), 
                       round(cm_dtree$byClass['F1'], 4), round(cm_rnf$byClass['F1'], 4))

model_results <- data.frame(Method, Accuracy, Balanced_Accuracy, F1_Score) # Create dataframe of the model results
#print out the results
model_results %>% knitr::kable()


