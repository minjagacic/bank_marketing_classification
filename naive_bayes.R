################################################
#             Naivni Bajes
################################################

# Ucitavanje dataseta, funckija i potrebnih biblioteka
bank.data <- readRDS("data/cleaned_bank_data.RData")
source("functions.R")

#install.packages("bnlearn")
library(e1071)
library(bnlearn)
library(caret)
library(ggplot2)
library(pROC)

# Naivni Bajes radi sa faktorskim promen. i numerickim sa N raspodelom
str(bank.data)
bank.data_nb <- bank.data

# Za sve numericke promenljive proveravamo da li imaju normalnu raspodelu
apply(X =bank.data_nb[,c(4,8:10)], MARGIN = 2, FUN = shapiro.test)

# p < 0.05 za sve promenljive sto znaci da nijedna nema N raspodelu
# Neophodna je diskretizacija za sve numericke promenljive

# int pretvaramo u numeric
int.vars <- c("balance","campaign","pdays","previous")
bank.data_nb[,int.vars] <- lapply(bank.data_nb[,int.vars], as.numeric)
str(bank.data_nb)

to.discretize <- bank.data_nb[,int.vars]
summary(to.discretize)

# Ispitujemo raspon promenljivih kako bismo odredili broj intervala
summary(bank.data_nb$balance)
ggplot(data = bank.data_nb, mapping = aes(x = balance)) + geom_histogram(bins = 20) 

summary(bank.data_nb$campaign)
ggplot(data = bank.data_nb, mapping = aes(x = campaign)) + geom_histogram(bins = 20)  

summary(bank.data_nb$pdays)
ggplot(data = bank.data_nb, mapping = aes(x = pdays)) + geom_histogram(bins = 20)  

summary(bank.data_nb$previous)
ggplot(data = bank.data_nb, mapping = aes(x = previous)) + geom_histogram(bins = 20) 

# pdays i previous ne mogu na vise od 1 intervala pa cemo ih pretvoriti u faktorske promenljive
# Pravimo binarnu klasifikaciju, sve vrednosti vece od 0 bice 1, u suprotnom bice 0

bank.data_nb$pdays <- ifelse(test = bank.data_nb$pdays > 0, yes = 1, no = 0)
bank.data_nb$previous <- ifelse(test = bank.data_nb$previous > 0, yes = 1, no = 0)

to_factor <- c("pdays","previous")
bank.data_nb[to_factor] <- lapply(bank.data_nb[to_factor], factor)
str(bank.data_nb)

# Ostale su nam samo dve numericke koje treba da diskretizujemo
to_discretize <- c("balance","campaign")

discretized <- discretize(data = bank.data_nb[,to_discretize],
                          method = "quantile",
                          breaks = c(3,2))
str(discretized)
summary(discretized)

bank.data_nb_new <- cbind(discretized, bank.data_nb[,c(1:3,5:7,9:12)])
bank.data_nb_new <- bank.data_nb_new[,names(bank.data_nb)]
str(bank.data_nb_new)

# Nakon diskretizacije dataset je spreman za kreiranje modela


# Podela dataseta na train i test
set.seed(123)
ind <- createDataPartition(bank.data_nb_new$y, p=0.8, list = F)
bank.train_nb <- bank.data_nb_new[ind,]
bank.test_nb <- bank.data_nb_new[-ind,]

# ------------   Kreiranje modela  ------------ #   

#?naiveBayes
nb1 <- naiveBayes(y ~ ., data = bank.train_nb)
print(nb1)

# Predikcije, matrica konfuzije i evaluacione metrike
nb1.pred <- predict(nb1, newdata = bank.test_nb, type = "class")
nb1.cm <- table(true = bank.test_nb$y, predicted = nb1.pred)
nb1.cm

nb1.eval <- compute.eval.metrics(nb1.cm)
nb1.eval
# prec = 0.289,  rec (sensitivity) = 0.202, specificity = 0.9,  F1 = 0.238

nb1.pred.prob <- predict(nb1, newdata = bank.test_nb, type = "raw")

nb1.roc <- roc(response = as.numeric(bank.test_nb$y), 
               predictor = nb1.pred.prob[,2],
               levels = c(1,2)) 

nb1.auc <- nb1.roc$auc 
nb1.auc #0.699


# ------------   Balansiranje podataka pri kros-validaciji ------------ #   

library(DMwR)
library(ROSE)
#install.packages("naivebayes")
library(naivebayes)

ctrl <- trainControl(method = "cv", number = 10,
                     classProbs = TRUE,
                     summaryFunction = twoClassSummary,
                     sampling = "down")

grid <- expand.grid(usekernel = TRUE, laplace = 1, adjust = c(0,0.5,1))

set.seed(5627)
down_inside <- train(x = bank.train_nb[,-12],
                     y = bank.train_nb$y,
                     method = "naive_bayes",
                     metric = "ROC",
                     tuneGrid = grid,
                     trControl = ctrl)

ctrl$sampling <- "up"
set.seed(5627)
up_inside <- train(x = bank.train_nb[,-12],
                   y = bank.train_nb$y,
                   method = "naive_bayes",
                   metric = "ROC",
                   tuneGrid = grid,
                   trControl = ctrl)

ctrl$sampling <- "rose"
set.seed(5627)
rose_inside <- train(x = bank.train_nb[,-12],
                     y = bank.train_nb$y,
                     method = "naive_bayes",
                     metric = "ROC",
                     tuneGrid = grid,
                     trControl = ctrl)

ctrl$sampling <- "smote"
set.seed(5627)
smote_inside <- train(x = bank.train_nb[,-12],
                      y = bank.train_nb$y,
                      method = "naive_bayes",
                      metric = "ROC",
                      tuneGrid = grid,
                      trControl = ctrl)

inside_models <- list(down = down_inside,
                      up = up_inside,
                      SMOTE = smote_inside,
                      ROSE = rose_inside)


#?resamples
inside_resampling <- resamples(inside_models) 

summary(inside_resampling, metric = "ROC")

#ROC 
#           Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
# down  0.6352533 0.6859016 0.7026844 0.7027323 0.7222541 0.7647751    0
# up    0.6298611 0.6862090 0.7145287 0.7094996 0.7403279 0.7660361    0
# SMOTE 0.6233252 0.6732889 0.6978893 0.6958495 0.7112500 0.7641866    0
# ROSE  0.6493873 0.6898975 0.7113320 0.7093856 0.7352049 0.7557797    0

# up-sampling daje najbolje rezultate



# ------------   Kreiranje modela sa balansiranim podacima ------------ #   

nb2.pred <- predict(object = up_inside, newdata = bank.test_nb, method = "class")

nb2.cm <- table(true = bank.test_nb$y, predicted = nb2.pred)
nb2.cm

nb2.eval <- compute.eval.metrics(nb2.cm)
nb2.eval
# prec = 0.206,  rec (sensitivity) = 0.565, specificity = 0.927,  F1 = 0.302

nb2.auc <- test_roc(model = up_inside, data = bank.test_nb) 
nb2.auc # 95% CI: 0.652-0.7577

nb2.roc <- roc(bank.test_nb$y,
               predict(up_inside, bank.test_nb, type = "prob")[, "yes"],
               levels = c("no", "yes"))

nb2.auc <- nb2.roc$auc 
nb2.auc #0.7048


# ------------   Odredjivanje optimalnog praga  ------------ #   

plot.roc(nb2.roc, print.thres = TRUE, print.thres.best.method = "youden") # 0.571(0.770, 0.535)
plot.roc(nb2.roc, print.thres = TRUE, print.thres.best.method = "closest.topleft") # 0.492(0.713, 0.586)

threshold <- 0.571

nb3.pred.prob <- predict(up_inside, newdata = bank.test_nb, type = "prob")

nb3.pred <- ifelse(test = nb3.pred.prob[,2] >= threshold, yes = "Yes", no = "No")
nb3.pred <- as.factor(nb3.pred)

nb3.cm <- table(true = bank.test_nb$y, predicted = nb3.pred)
nb3.cm

nb3.eval <- compute.eval.metrics(nb3.cm)
nb3.eval
# prec = 0.232,  rec (sensitivity) = 0.535, specificity = 0.927,  F1 = 0.324

nb3.auc <- nb2.auc # 0.7048

nb.coords <- coords(nb2.roc, ret = c("spec","sens","thr"),x = "local maximas", transpose = FALSE)
nb.coords

# Cilj nam je da maksimizujemo vrednosti za sensitivity te biramo:
# specificity   sensitivity   threshold
# 0.41601050    0.85858586    0.25414093

threshold <- 0.25414093

nb4.pred <- ifelse(test = nb3.pred.prob[,2] >= threshold, yes = "Yes", no = "No")
nb4.pred <- as.factor(nb4.pred)

nb4.cm <- table(true = bank.test_nb$y, predicted = nb4.pred)
nb4.cm

nb4.eval <- compute.eval.metrics(nb4.cm)
nb4.eval
# prec = 0.160,  rec (sensitivity) = 0.858, specificity = 0.957,  F1 = 0.270

nb4.auc <- nb2.auc  # 0.7048

# ------------   Poredjenje rezultata  ------------ # 

AUC <- c(nb1.auc, nb2.auc, nb3.auc, nb4.auc)
df_nb_metrics <- data.frame(rbind(nb1.eval, nb2.eval, nb3.eval, nb4.eval), row.names = c("nb 1 ","nb 2 ", "nb 3 ", "nb 4 "))
df_nb_metrics <- cbind(df_nb_metrics, AUC)
df_nb_metrics

#       precision    recall specificity        F1       AUC
# nb 1  0.2898551 0.2020202   0.9002525 0.2380952 0.6998794
# nb 2  0.2066421 0.5656566   0.9271186 0.3027027 0.7048371
# nb 3  0.2324561 0.5353535   0.9273302 0.3241590 0.7048371
# nb 4  0.1603774 0.8585859   0.9577039 0.2702703 0.7048371

# nb1 je model kreiran sa podrazumevanim vrednostima i u skladu sa tim, rezultati nisu dovoljno dobri, tacnije
#       vrednost sensitivity (recall) je dosta niska

# nb2 je model kreiran sa balansiram podacima te su rezultati bolji u odnosu na prvi model, 
#       vidimo rast sensitivity metrike; takodje, ovu vrednost AUC metrike koristimo i u sledecim modelima

# nb3 je model kreiran sa balansiranim podacima i biranom threshold vrednosti metodama "youden" i "closest top-left",
#       ipak, sensitivity (recall) metrika je nesto niza u odnosu na nb2 model

# nb4 je model kreiran sa balansiranim podacima i biranim threshold vrednosti tako da se maksimizira sensitivity 
#       (recall) metrika, ali pazeci da vrednost za specificity ne bude previse niska.

# Nas cilj je da sto tacnije predvidimo pozitivnu, "yes" klasu, te nam je bitno da sensitivity metrika bude sto veca.
# Zato biramo nb4 model kao model koji daje najbolje rezultate.
