################################################
#                 Funkcije
################################################

# Cesto koriscene funkcije cuvamo u jednom fajlu i pozivamo ih odatle

# Funkcija za izracunavanje evaluacionih metrika
compute.eval.metrics <- function(cmatrix){
  
  TP <- cmatrix[2,2]
  TN <- cmatrix[1,1]
  FP <- cmatrix[1,2]
  FN <- cmatrix[2,1]
 
  prec <- TP / (TP + FP)
  rec <- TP / (TP + FN)
  spec <- TN / (TN + FN)
  F1 <- 2*prec*rec / (prec + rec)
  
  c(precision = prec, recall = rec, specificity = spec, F1 = F1)
  
  # pozitivna klasa nam je YES
  # TP - true positive, predvidjene kao pozitivne i stvarno pozitivne
  # TN - true negative, predvidjene kao negativne i stvarno negativne
  # FP - false positive, predvidjene kao pozitivne a stvarno negativne
  # FN - false negative, predvidjene kao negativne a stvarno pozitivne
  
  # precision - preciznost modela,  stvarno pozitivne od svih predvidjenih kao pozitivne
  # recall (sensitivity) - odziv modela, broj stvarno pozitivnih od svih ukupno pozitivnih
  # specificity - broj stvarno negativnih od svih ukupno negativnih
  # F1 - ocena modela
}
  
test_roc <- function(model, data) {
  roc_obj <- roc(data$y, 
                 predict(model, data, type = "prob")[, "yes"],
                 levels = c("no", "yes")) #pravimo ROC krivu
  ci(roc_obj) # interval poverenja za ROC krivu
}
