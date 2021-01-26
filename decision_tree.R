################################################
#              Drvo odlucivanja
################################################

# Ucitavanje sredjenog skupa podataka
bank.data <- readRDS("data/cleaned_bank_data.RData")
str(bank.data)
summary(bank.data)

table(bank.data$y)
# Za pozitivnu klasu uzimamo vrednosti "yes"

# Funkcije koje koristimo u svim modelima cuvamo u jednom fajlu i pozivamo ih odatle
# Ucitavanje funkcija
source("functions.R")

#install.packages("pROC")
library(pROC)
library(ROSE)

# Podela dataseta na train i test
library(lattice)
library(caret)

set.seed(123)
ind <- createDataPartition(bank.data$y, p=0.8, list = F)
bank.train <- bank.data[ind,]
bank.test <- bank.data[-ind,]

# Provera raspodele, cilj nam je da bude ujednaceno u oba dataseta
prop.table(table(bank.train$y))
prop.table(table(bank.test$y))

# ------------   Kreiranje modela sa default vrednostima ------------ #   
#  default complexity parametar cp = 0.001

library(rpart)
library(rpart.plot)

#?rpart
#?rpart.control

# Kreiranje stabla
set.seed(123)
tree1 <- rpart(y ~ ., data = bank.train, method = "class")
rpart.plot(tree1)

# Predikcije
tree1.pred <- predict (object = tree1, newdata = bank.test, type = "class")
cbind(head(tree1.pred),
      head(bank.test$y))

# Matrica konfuzije
tree1.cm <- table(actual = bank.test$y, predicted = tree1.pred)
tree1.cm

# ROC kriva
tree1.pred.prob <- predict (object = tree1, newdata = bank.test)
tree1.auc <- roc.curve(bank.test$y, tree1.pred.prob[,2])$auc 
tree1.auc  # 0.566

# Evaluacione metrike
tree1.eval <- compute.eval.metrics(tree1.cm)
tree1.eval
# prec = 0.667,  rec (sensitivity) = 0.141, specificity = 0.898,  F1 = 0.233
# Model nije dobar


# ---------------    Kros-validacija    ---------------   #
# Da bismo odredili optimalnu vrednost parametra kompleksnosti (cp) radicemo kros-validaciju
# radicemo kros-validaciju sa 10 iteracija

library(e1071)

numFolds <- trainControl(method = "cv", 
                         number = 10,
                         classProbs = TRUE,
                         summaryFunction = twoClassSummary)

cpGrid <- expand.grid(.cp = seq(from = 0.001, to = 0.05, by = 0.001))

set.seed(123)
tree.cv <- train (x = bank.train[,-12],
                   y = bank.train$y,
                   method = "rpart", 
                   trControl = numFolds, 
                   tuneGrid = cpGrid,
                   metric = "ROC")
tree.cv
plot(tree.cv)

cp.opt <- tree.cv$bestTune$cp
cp.opt
# Optimalna vrednost parametra cp je 0.001


# ------------   Model sa optimalnim cp ------------ # 

# Nakon kros-validacije dobili smo da je optimalna vrednost cp = 0.001
#  sto je i default vrednost za ovaj parametar
# Ipak, napravicemo jos jedan (pokazni) model sa optimalnim cp

tree2 <- prune(tree1, cp = cp.opt) 
rpart.plot(tree2)
tree2.pred <- predict(tree2, newdata = bank.test, type = "class")
tree2.cm <- table(true = bank.test$y, predicted = tree2.pred)
tree2.cm
tree2.eval <- compute.eval.metrics(tree2.cm)
tree2.eval

tree2.pred.prob <- predict (object = tree2, newdata = bank.test)
tree2.auc <- roc.curve(bank.test$y, tree2.pred.prob[,2])$auc 
tree2.auc # 0.5661
# Ocekivano dobijamo iste vrednosti kao i za prethodni model 

# S obzirom na to da dataset nije uravnotezen, moramo da uradimo balansiranje podataka

# ------------   Balansiranje podataka pri kros-validaciji ------------ #   

library(gplots)
#install.packages("DMwR")
library(DMwR)
library(ROSE)

ctrl <- trainControl(method = "cv", number = 10,
                     classProbs = TRUE,
                     summaryFunction = twoClassSummary,
                     sampling = "down")

cpGrid <- expand.grid(.cp = seq(from = 0.001, to = 0.05, by = 0.001))

set.seed(5627)
down_inside <- train(x = bank.train[,-12],
                     y = bank.train$y,
                     method = "rpart",
                     metric = "ROC",
                     tuneGrid = cpGrid,
                     trControl = ctrl)

ctrl$sampling <- "up"
set.seed(5627)
up_inside <- train(x = bank.train[,-12],
                   y = bank.train$y,
                   method = "rpart",
                   metric = "ROC",
                   tuneGrid = cpGrid,
                   trControl = ctrl)

ctrl$sampling <- "rose"
set.seed(5627)
rose_inside <- train(x = bank.train[,-12],
                     y = bank.train$y,
                     method = "rpart",
                     metric = "ROC",
                     tuneGrid = cpGrid,
                     trControl = ctrl)

ctrl$sampling <- "smote"
set.seed(5627)
smote_inside <- train(x = bank.train[,-12],
                      y = bank.train$y,
                      method = "rpart",
                      metric = "ROC",
                      tuneGrid = cpGrid,
                      trControl = ctrl)

inside_models <- list(down = down_inside,
                      up = up_inside,
                      SMOTE = smote_inside,
                      ROSE = rose_inside)


#?resamples
inside_resampling <- resamples(inside_models) 

summary(inside_resampling, metric = "ROC")

#  ROC 
#             Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
# down  0.6056557 0.6334335 0.6756762 0.6791887 0.7254508 0.7622951    0
# up    0.6250000 0.6476134 0.6643033 0.6830500 0.7228484 0.7630738    0
# SMOTE 0.6115164 0.6300410 0.6611185 0.6593201 0.6818135 0.7106768    0
# ROSE  0.5173203 0.6228279 0.6543443 0.6509941 0.6979098 0.7523749    0

# up-sampling daje najbolje rezultate


# -------    Model sa balansiranim podacima ------- #

tree3.pred <- predict(object = up_inside, newdata = bank.test, method = "class")
tree3.cm <- table(true = bank.test$y, predicted = tree3.pred)
tree3.cm
tree3.eval <- compute.eval.metrics(tree3.cm)
tree3.eval
# prec = 0.23,  rec (sensitivity) = 0.484, specificity = 0.921,  F1 = 0.313

tree3.auc <- test_roc(model = up_inside, data = bank.test) 
tree3.auc # 95% CI: 0.6292-0.7418

tree3.auc <- roc(bank.test$y,
                 predict(up_inside, bank.test, type = "prob")[, "yes"],
                 levels = c("no", "yes"))$auc
tree3.auc # 0.6855

?varImp
tree3.Imp <- varImp(up_inside, scale = TRUE)
tree3.Imp


# -------   Poredjenje metrika ------- #

AUC <- c(tree1.auc, tree2.auc, tree3.auc)
df_trees_metrics <- data.frame(rbind(tree1.eval, tree2.eval, tree3.eval), row.names = c("tree 1 ","tree 2 ", "tree 3"))
df_trees_metrics <- cbind(df_trees_metrics, AUC)
df_trees_metrics

# Poredjenjem sva tri modela zakljucujemo sledece:
# tree1 i tree2 imaju iste vrednosti evaluacionih metrika jer rade sa istim podacima
# tree3 radi sa balansiranim podacima i to podacima balansiranim up-sampling tehnikom
# tree3 model povecava vrednosti metrika koje su nama vazne: sensitivity, specificity i  AUC 

