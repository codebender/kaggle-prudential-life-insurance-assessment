# Load Dependeicies
library(caret)
library(xgboost)
library(readr)

# Set seed
set.seed(1337)

# Load data
train <- read_csv("../input/train.csv")
test  <- read_csv("../input/test.csv")

# Categorical Variables
# catVars <- c('Product_Info_1', 'Product_Info_2', 'Product_Info_3',
#              'Product_Info_5', 'Product_Info_6', 'Product_Info_7',
#              'Employment_Info_2', 'Employment_Info_3', 'Employment_Info_5',
#              'InsuredInfo_1', 'InsuredInfo_2', 'InsuredInfo_3', 'InsuredInfo_4',
#              'InsuredInfo_5', 'InsuredInfo_6', 'InsuredInfo_7',
#              'Insurance_History_1', 'Insurance_History_2',
#              'Insurance_History_3', 'Insurance_History_4',
#              'Insurance_History_7', 'Insurance_History_8',
#              'Insurance_History_9', 'Family_Hist_1', 'Medical_History_2',
#              'Medical_History_3', 'Medical_History_4', 'Medical_History_5',
#              'Medical_History_6', 'Medical_History_7', 'Medical_History_8',
#              'Medical_History_9', 'Medical_History_11',
#              'Medical_History_12', 'Medical_History_13', 'Medical_History_14',
#              'Medical_History_16', 'Medical_History_17', 'Medical_History_18',
#              'Medical_History_19', 'Medical_History_20', 'Medical_History_21',
#              'Medical_History_22', 'Medical_History_23', 'Medical_History_25',
#              'Medical_History_26', 'Medical_History_27', 'Medical_History_28',
#              'Medical_History_29', 'Medical_History_30', 'Medical_History_31',
#              'Medical_History_33', 'Medical_History_34', 'Medical_History_35',
#              'Medical_History_36', 'Medical_History_37', 'Medical_History_38',
#              'Medical_History_39', 'Medical_History_40', 'Medical_History_41')
#
# for (var in catVars) {
#   train[[var]] <- factor(train[[var]])
#   test[[var]] <- factor(test[[var]])
# }


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

tra<-train[,feature.names]
h<-sample(nrow(train),12000)
dval<-xgb.DMatrix(data=data.matrix(tra[h,]),label=train$Response[h], missing = NA)
dtrain<-xgb.DMatrix(data=data.matrix(tra[-h,]),label=train$Response[-h], missing = NA)
watchlist<-list(val=dval,train=dtrain)

param <- list(  objective           = "reg:linear",
                eta                 = 0.025,
                depth               = 20,
                subsample           = 0.7,
                colsample_bytree    = 0.7,
                min_child_weight    = 3
)

clf <- xgb.train(   params              = param,
                    data                = dtrain,
                    nrounds             = 1000,
                    verbose             = 0,
                    early.stop.round    = 50,
                    watchlist           = watchlist,
                    maximize            = FALSE,
                    eval_metric         = 'rmse'
)

importance_matrix <- xgb.importance(feature.names, model = clf)
xgb.plot.importance(importance_matrix)

clf <- xgboost(data        = data.matrix(train[,feature.names]),
               label       = train$Response,
               eta         = 0.025,
               depth       = 25,
               nrounds     = 4000,
               objective   = "reg:linear",
               eval_metric = "rmse",
               colsample_bytree=0.7,
               min_child_weight=3,
               subsample=0.7)

importance_matrix <- xgb.importance(feature.names, model = clf)
xgb.plot.importance(importance_matrix[1:20,])

submission <- data.frame(Id=test$Id)
submission$Response <- as.integer(round(predict(clf, data.matrix(test[,feature.names]))))

submission[submission$Response < 1, "Response"] <- 1
submission[submission$Response > 8, "Response"] <- 8

write_csv(submission, "../submissions/xgboost_starter_3.csv")
