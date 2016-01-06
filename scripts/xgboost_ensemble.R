# Load Dependeicies
library(readr)
library(xgboost)

# Set seed
set.seed(1337)

# Load data
train <- read_csv("../input/train.csv")
test  <- read_csv("../input/test.csv")

# Dummy Variables
dummyVars <- paste('Medical_Keyword_', 1:48, sep = '')

for (var in dummyVars) {
  train[[var]] <- factor(train[[var]])
  test[[var]] <- factor(test[[var]])
}

feature.names <- names(train)[2:(ncol(train)-1)]

# Fix NA's
for (f in feature.names) {
  if (class(train[[f]])=="integer" || class(train[[f]])=="numeric") {
    mean <- mean(train[[f]], na.rm = T)
    train[[f]][is.na(train[[f]])] <- mean
    test[[f]][is.na(test[[f]])] <- mean
  }
}

# Convert character columns to ids
for (f in feature.names) {
  if (class(train[[f]])=="character") {
    levels <- unique(c(train[[f]], test[[f]]))
    train[[f]] <- as.integer(factor(train[[f]], levels=levels))
    test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))
  }
}

y = train$Response
y = as.integer(y)-1

clfSoft <- xgboost(data        = data.matrix(train[,feature.names]),
                   label       = y,
                   eta         = 0.02,
                   depth       = 21,
                   nrounds     = 4500,
                   objective   = "multi:softmax",
                   eval_metric = "mlogloss",
                   num_class   = 8,
                   colsample_bytree = 0.65,
                   min_child_weight = 3,
                   subsample = 0.73)

predictions <- data.frame(Id=test$Id)
predictions$ResponseSoft <- predict(clfSoft, data.matrix(test[,feature.names])) + 1

clfLinear <- xgboost(data        = data.matrix(train[,feature.names]),
                     label       = train$Response,
                     eta         = 0.02,
                     depth       = 21,
                     nrounds     = 4500,
                     objective   = "reg:linear",
                     eval_metric = "rmse",
                     colsample_bytree = 0.65,
                     min_child_weight = 3,
                     subsample = 0.73)
predictions$ResponseLinear <- as.integer(round(predict(clfLinear, data.matrix(test[,feature.names]))))
predictions[predictions$ResponseLinear < 1, "ResponseLinear"] <- 1
predictions[predictions$ResponseLinear > 8, "ResponseLinear"] <- 8

submission <- data.frame(Id=test$Id)
submission$Response <- as.integer(round((0.7*predictions$ResponseLinear+ 0.3*predictions$ResponseSoft)))

submission[submission$Response==3,"Response"] <- 2

write_csv(submission, "../submissions/xgboost_ensemble_6.csv")
