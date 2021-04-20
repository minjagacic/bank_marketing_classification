################################################
#            Random forest
################################################

# Ucitavanje dataseta, funckija i potrebnih biblioteka
bank.data <- readRDS("data/cleaned_bank_data.RData")
source("functions.R")

library(lattice)
library(ggplot2)
library(caret)
library(pROC)
#install.packages("randomForest")
library(randomForest)

?randomForest

# Ispitivanje strukture podataka
str(bank.data)
summary(bank.data)
table(bank.data$y)

# Algoritam Random forest radi i sa numerickim i sa kategorickim podacima,
#   sto znaci da nije potrebno dodatno sredjivanje dataseta

# Podela na train i test
set.seed(123)
ind <- createDataPartition(bank.data$y, p=0.8, list = F)
bank.train <- bank.data[ind,]
bank.test <- bank.data[-ind,]


# ------------   Kreiranje modela sa default vrednostima ------------ #   
# ntree - broj stabala u modelu, default ntree = 500
# mtry - broj promenljivih koje se nasumicno biraju pri grananju stabla, mtry = sqrt(n) gde je n broj varijabli

set.seed(222)
rf1 <- randomForest(y ~., data = bank.train, importance = TRUE)

print(rf1)
# ntree = 500, mtry =  3
# OOB error = 11.36%

importance(rf1)
varImpPlot(rf1)

# Predikcije
rf1.pred <- predict(rf1, newdata = bank.test, type = "class")

rf1.cm <- table(true = bank.test$y, predicted = rf1.pred)
rf1.cm 

rf1.eval <- compute.eval.metrics(rf1.cm)
rf1.eval
# prec = 0.516,  rec (sensitivity) = 0.161, specificity = 0.98,  F1 = 0.246

rf1.pred.prob <- predict(object = rf1, newdata = bank.test,type="prob")

rf1.roc <-roc(response = as.numeric(bank.test$y),
              predictor = rf1.pred.prob[,2] ,
              levels = c(1, 2))

rf1.auc <- rf1.roc$auc 
rf1.auc  # 0.6911


# ------------   "Tuning" modela   ------------ #   
# Odredjivanje optimalnih vrednosti parametara 

set.seed(123)
tunemtry <-tuneRF(bank.train[,-12],
                  bank.train[,12],
                  stepFactor = 2,   
                  plot = TRUE,
                  ntreeTry = 1500, 
                  trace = TRUE,
                  improve = 0.01)

print(tunemtry)
# mtry = 2 	OOB error = 10.99% 
bestmtry <- 2

# povecanjem parametra ntree smanjujemo OOB gresku
bestntree <- 1500


# Drugi model kreiramo sa ovim parametrima
rf2 <- randomForest(y ~ ., 
                    data = bank.train, 
                    mtry = bestmtry, ntree = bestntree, importance = TRUE) 
print(rf2)
# ntree = 1500,  mtry = 2,  OOB = 10.99%

varImpPlot(rf2)
importance(rf2)

rf2.pred <- predict(object = rf2, newdata = bank.test, method = "class")

rf2.cm <- table(true = bank.test$y, predicted = rf2.pred)
rf2.cm

rf2.eval <- compute.eval.metrics(rf2.cm)
rf2.eval
# prec = 0.636,  rec (sensitivity) = 0.141, specificity = 0.989,  F1 = 0.231

rf2.pred.prob <- predict(object = rf2, newdata = bank.test,type="prob")

rf2.roc <-roc(response = as.numeric(bank.test$y),
              predictor = rf2.pred.prob[,2] ,
              levels = c(1, 2))

rf2.auc <- rf2.roc$auc 
rf2.auc  # 0.7009


# ------------   Balansiranje podataka pri kros-validaciji ------------ #   

library(DMwR)
library(ROSE)

ctrl <- trainControl(method="repeatedcv", 
                     number=10, 
                     repeats=3, 
                     search="grid",
                     classProbs = TRUE,
                     summaryFunction = twoClassSummary,
                     sampling = "down")

set.seed(5672)

grid <- expand.grid(.mtry=c(1:10))

down_inside <- train(x = bank.train[,-12],
                     y = bank.train$y,
                     method = "rf",
                     metric = "ROC",
                     tuneGrid = grid,
                     trControl = ctrl,
                     ntree = 1500)

print(down_inside)
plot(down_inside)
# za down-sampling optimalan mtry = 1

ctrl$sampling <- "up"
set.seed(5627)

up_inside <- train(x = bank.train[,-12],
                   y = bank.train$y,
                   method = "rf",
                   metric = "ROC",
                   tuneGrid = grid,
                   trControl = ctrl,
                   ntree = 1500)

print(up_inside)
plot(up_inside)
# za up-sampling optimalan mtry = 1

ctrl$sampling <- "rose"
set.seed(5627)

rose_inside <- train(x = bank.train[,-12],
                     y = bank.train$y,
                     method = "rf",
                     metric = "ROC",
                     tuneGrid = grid,
                     trControl = ctrl,
                     ntree = 1500)

print(rose_inside)
plot(rose_inside)
# za rose-sampling optimalan mtry = 1

ctrl$sampling <- "smote"
set.seed(5627)

smote_inside <- train(x = bank.train[,-12],
                     y = bank.train$y,
                     method = "rf",
                     metric = "ROC",
                     tuneGrid = grid,
                     trControl = ctrl,
                     ntree = 1500)

print(smote_inside)
plot(smote_inside)
# za smote-sampling optimalan mtry = 3

inside_models <- list(down = down_inside,
                      up = up_inside,
                      SMOTE = smote_inside,
                      ROSE = rose_inside)


inside_resampling <- resamples(inside_models) 

summary(inside_resampling, metric = "ROC")

#ROC 
#           Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
# down  0.6395082 0.6875044 0.7223975 0.7107552 0.7381250 0.7789344    0
# up    0.6680964 0.6906455 0.7151434 0.7220736 0.7485041 0.8172541    0
# SMOTE 0.6312500 0.6602766 0.6883542 0.6932673 0.7175911 0.7961066    0
# ROSE  0.6491013 0.6849283 0.7045902 0.7119259 0.7360143 0.8023361    0

# up-sampling daje najbolje rezultate, sto znaci da koristimo mtry = 1



# ------------   Kreiranje modela sa balansiranim podacima ------------ # 

rf3.pred <- predict(object = up_inside, newdata = bank.test, method = "class")

rf3.cm <- table(true = bank.test$y, predicted = rf3.pred)
rf3.cm

rf3.eval <- compute.eval.metrics(rf3.cm)
rf3.eval
# prec = 0.281,  rec (sensitivity) = 0.454, specificity = 0.849,  F1 = 0.347

rf3.pred.prob <- predict(object = up_inside, newdata = bank.test,type="prob")

rf3.roc <-roc(response = as.numeric(bank.test$y),
              predictor = rf3.pred.prob[,2] ,
              levels = c(1, 2))

rf3.auc <- rf3.roc$auc 
rf3.auc  # 0.7091


# ------------   Poredjenje rezultata  ------------ # 

AUC <- c(rf1.auc, rf2.auc, rf3.auc)
df_rf_metrics <- data.frame(rbind(rf1.eval, rf2.eval, rf3.eval), row.names = c("rf 1 ","rf 2 ", "rf 3 "))
df_rf_metrics <- cbind(df_rf_metrics, AUC)
df_rf_metrics

#       precision    recall     specificity   F1          AUC
# rf 1  0.5161290 0.1616162   0.9803150 0.2461538   0.6911305
# rf 2  0.6363636 0.1414141   0.9895013 0.2314050   0.7009001
# rf 3  0.2812500 0.4545455   0.8490814 0.3474903   0.7090790

# rf1 je model kreiran sa default parametrima ntree = 500 i mtry = 3
# rf2 je model kreiran nakon tuning-a i parametri koji su korisceni su ntree = 1500 i mtry = 2
# rf3 je model kreiran nakon balansiranja dataseta tokom kog smo radili i tuning te su korisceni ntree = 1500, mtry = 1

# Nakon prvog tuninga modela, preciznost, specificity i AUC metrike su se povecale, dok su recall i F1 samo opale
# Balansiranjem dataseta, metrike su se znacajnije promenile: preciznost je znacajno opala dok se recall povecao;
#     specificity je opao, a F1 i AUC su se malo povecale

# Imajuci u vidu da je nama cilj maksimiziranje recall metrike, 
#       model kreiran nad balansiranim podacima se smatra najboljim 


