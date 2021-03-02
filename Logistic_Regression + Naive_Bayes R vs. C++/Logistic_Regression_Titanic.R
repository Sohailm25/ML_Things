# Logistic_Regression_Titanic


# Training a Logistic Regression Model
titanic <- read.csv('titanic_project.csv')
attach(titanic)

# split train and test Use the first 900 observations for train, the rest for test.  
train <- titanic[1:900,]
test <- titanic[901:1046,]

# start timer, train logistic regression model, and end timer
start_time <- Sys.time()
logRegModel <- glm(survived~pclass, data=train, family=binomial)
end_time <- Sys.time()
runtime <- end_time - start_time
print(paste("TRAINING LOGREGMODEL RUNTIME: ", runtime, "seconds"))

# print logistic regression model coefficients
print("MODEL COEFFICIENTS: ")
print(logRegModel$coefficients)

probs <- predict(logRegModel, newdata = test, type="response")
pred <- ifelse(probs > 0.5, 1, 0)
print("confusion Matrix:")
print(table(pred, test$survived))

# Metrics

TP <- sum(pred == 1 & test$survived == 1)
FP <- sum(pred == 1 & test$survived == 0)
FN <- sum(pred == 0 & test$survived == 1)
TN <- sum(pred == 0 & test$survived == 0)

accuracy <- (TP + TN) / (TP + TN + FP + FN)
sensitivity <- (TP) / (TP + FN)
specificity <- (TN) / (TN + FP)

## Accuracy
print(paste("ACCURACY:", accuracy))

## Sensitivity
print(paste("SENSITIVITY:", sensitivity))

## Specificity
print(paste("SPECIFICITY:", specificity))


