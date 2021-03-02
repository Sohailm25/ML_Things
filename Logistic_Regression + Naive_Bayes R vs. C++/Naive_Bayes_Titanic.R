
# Training a Naive Bayes Model

titanic <- read.csv('titanic_project.csv')

# Use the first 900 observations for train, the rest for test.  
attach(titanic)
titanic$pclass <- as.factor(titanic$pclass)
titanic$sex <- as.factor(titanic$sex)
train <- titanic[1:900,]
test <- titanic[901:1046,]

# Naive Bayes
library(e1071)
start_time <- Sys.time()
naiveBayesModel <- naiveBayes(survived ~ pclass+sex +age, data=train)
end_time <- Sys.time()
runtime <- end_time - start_time
print(paste("TRAINING NAIVEBAYES MODEL RUNTIME:", runtime))

print("naiveBayesModel:")
print(naiveBayesModel)

# Test on test data
raw <- predict(naiveBayesModel, newdata = test, type="raw")
pred <- ifelse(raw[,2] > 0.5, 1, 0)
pred <- as.factor(pred)

if(!require('caret')) {
  install.packages('caret', dependencies = TRUE)
}
library(caret)

print("Confusion Matrix")
print(confusionMatrix(pred, as.factor(test$survived)))


# Metrics=

TP <- sum(pred == 1 & test$survived == 1)
FP <- sum(pred == 1 & test$survived == 0)
FN <- sum(pred == 0 & test$survived == 1)
TN <- sum(pred == 0 & test$survived == 0)

accuracy <- (TP + TN) / (TP + TN + FP + FN)
sensitivity <- (TP) / (TP + FN)
specificity <- (TN) / (TN + FP)

print(paste("ACCURACY", accuracy))
print(paste("SENSITIVITY", sensitivity))
print(paste("SPECIFICITY", specificity))


