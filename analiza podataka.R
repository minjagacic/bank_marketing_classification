################################################
#              Analiza podataka
################################################

# Ucitavanje dataseta
bank <- read.csv("data/bank.csv", stringsAsFactors = FALSE)

# Ispitivanje strukture
str(bank)
summary(bank)

# sve character promenljive prebacujmo u faktorske
bank$job <- as.factor(bank$job)
bank$marital <- as.factor(bank$marital)
bank$education <- as.factor(bank$education)
bank$default <- as.factor(bank$default)
bank$housing <- as.factor(bank$housing)
bank$loan <- as.factor(bank$loan)
bank$contact <- as.factor(bank$contact)
bank$month <- as.factor(bank$month)
bank$poutcome <- as.factor(bank$poutcome)
bank$y <- as.factor(bank$y)

str(bank)
summary(bank)
# izlazna varijabla Y nije uravnotezena, 4000 no i 521 yes

# Sortiranje faktorske promenljive month
levels(bank$month)
bank$month <- factor(bank$month, levels = c("jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"))
levels(bank$month)


# Provera nedostajucih vrednosti
apply(bank, 2, FUN = function(x)length(which(is.na(x))))
apply(bank, 2, FUN = function(x)length(which(x=="unknown")))

# Nema NA vrednosti
# ali postoje unknown vrednosti i to za varijable: job, education, contact i poutcome (sve su factor)

# Detaljnijom analizom dataseta utvrdjujemo da varijabla Contact, koja predstavlja sredstvo kontaktiranja klijenta,
#   nije od velikog znacaja za izlaznu promenljivu te cemo je iskljuciti iz dalje analize
bank$contact <- NULL
 
# Kada je rec o nepoznatim vrednostima za varijable Job i Education, dataset je dovoljno veliki da ove observacije  
#  mozemo izostaviti iz analize 

job.edu.unknown.ind <- which(bank$job == "unknown" | bank$education =="unknown")
bank <- bank[-job.edu.unknown.ind,]

table(bank$job)
bank$job <- factor(bank$job)
table(bank$job)

table(bank$education)
bank$education <- factor(bank$education)
table(bank$education)

# Varijabla Poutcome se odnosi na ishod kampanje i ukoliko je vrednost nepoznata to znaci da klijent nije bio deo 
#  prethodne kampanje i ishod ovih varijabli zapravo ne postoji. 
#  Zato cemo ove nepoznate vrednosti preimenovati u "nonexistent" i dodati joj kategoriju "other"

bank$poutcome <- as.character(bank$poutcome)
bank$poutcome[bank$poutcome %in% c('unknown', 'other')] <- "nonexistent"
bank$poutcome <- factor(bank$poutcome)
table(bank$poutcome)


apply(bank, 2, FUN = function(x)length(which(x=="unknown")))
# Uspesno smo uklonili sve "unknown" vrednosti


# Ispitivanje znacajnosti promenljivih u odnosu na izlaznu

# H0 pretpostavka je da su dve promenljive nezavisne, H1 da su zavisne
# Ukoliko je p vrednost manja od 0.05 odbacujemo H0 hipotezu, odnosno promenljive su zavisne
# U suprotnom atribut iskljcujemo iz analize jer nema znacajnost

str(bank)
summary(bank)

# Ispitivanje za faktorske promenljive (asocijacija) - koristimo bar plot grafikone i Chi square test

library(ggplot2)

# Uticaj promenljive JOB
ggplot(data = bank, mapping = aes(x = job, fill = y)) +
  geom_bar(position = "dodge", width = 0.5) +
  ylab("Number of phone calls") +
  xlab("Job") + 
  theme_bw()

tb1 <- table(bank$y,bank$job)
tb1
chisq.test(tb1)
# p = 1.502e-11 < 0.05  

# Uticaj promenljive MARITAL
ggplot(data = bank, mapping = aes(x = marital, fill = y)) +
  geom_bar(position = "dodge", width = 0.5) +
  ylab("Number of phone calls") +
  xlab("Marital status") +
  theme_bw()

tb2 <- table(bank$y,bank$marital)
tb2
chisq.test(tb2)
# p =1.699e-05 < 0.05 

# Uticaj promenljive EDUCATION
ggplot(data = bank, mapping = aes(x = education, fill = y)) +
  geom_bar(position = "dodge", width = 0.5) +
  ylab("Number of phone calls") +
  xlab("Education level") +
  theme_bw()

tb3 <- table(bank$y,bank$education)
tb3
chisq.test(tb3)
# p = 0.0004434 < 0.05 

# Uticaj promenljive DEFAULT
ggplot(data = bank, mapping = aes(x = default, fill = y)) +
  geom_bar(position = "dodge", width = 0.5) +
  ylab("Number of phone calls") +
  xlab("Default level") +
  theme_bw()

tb4 <- table(bank$y,bank$default)
tb4
chisq.test(tb4)
# p = 1 > 0.05 -> promenljive su nezavisne, mozemo je iskljuciti iz analize
bank$default <- NULL

# Uticaj promenljive HOUSING
ggplot(data = bank, mapping = aes(x = housing, fill = y)) +
  geom_bar(position = "dodge", width = 0.5) +
  ylab("Number of phone calls") +
  xlab("Housing") +
  theme_bw()

tb5 <- table(bank$y,bank$housing)
tb5
chisq.test(tb5)
# p = 5.949e-12 < 0.05 

# Uticaj promenljive LOAN
ggplot(data = bank, mapping = aes(x = loan, fill = y)) +
  geom_bar(position = "dodge", width = 0.5) +
  ylab("Number of phone calls") +
  xlab("Loan") +
  theme_bw()

tb6 <- table(bank$y,bank$loan)
tb6
chisq.test(tb6)
# p = 2.037e-06 < 0.05 


# Uticaj promenljive MONTH
ggplot(data = bank, mapping = aes(x = month, fill = y)) +
  geom_bar(position = "dodge", width = 0.5) +
  ylab("Number of phone calls") +
  xlab("Month") +
  theme_bw()

tb7 <- table(bank$y,bank$month)
tb7
chisq.test(tb7)
# p = 2.2e-16 < 0.05  

# Uticaj promenljive POUTCOME
ggplot(data = bank, mapping = aes(x = poutcome, fill = y)) +
  geom_bar(position = "dodge", width = 0.5) +
  ylab("Number of phone calls") +
  xlab("Poutcome") +
  theme_bw()

tb8 <- table(bank$y,bank$poutcome)
t8
chisq.test(tb8)
# p = 2.2e-16 < 0.05 


# Ispitivanje za numericke promenljive (korelacija)
# koristimo grafikone gustine raspodele i Kruskal-Wallis test

ggplot(data = bank, aes(x = age, fill = y)) + geom_density(alpha = 0.5)+ theme_bw()
kruskal.test(age ~ y, data = bank) 
# p = 0.5633 > 0.05 -> promenljive su nezavisne
bank$age <- NULL

ggplot(data = bank, aes(x = balance, fill = y)) + geom_density(alpha = 0.5) + xlim(-3000,10000) + theme_bw()
kruskal.test(balance ~ y, data = bank)
# p = 5.731e-07 < 0.05

ggplot(data = bank, aes(x = day, fill = y)) + geom_density(alpha = 0.5)+ theme_bw()
kruskal.test(day ~ y, data = bank)
# p = 0.482 > 0.05 -> promenljive su nezavisne
bank$day <- NULL

ggplot(data = bank, aes(x = duration, fill = y)) + geom_density(alpha = 0.5) + xlim(0,2500)+ theme_bw()
kruskal.test(duration ~ y, data = bank)
# p = 2.2e-16 < 0.05
# Iako p vrednost sugerise da je promenljiva znacajna, u opisu stoji da ne treba da se koristi za prediktive modele
#  te cemo je iskljuciti
bank$duration <- NULL

ggplot(data = bank, aes(x = campaign , fill = y)) + geom_density(alpha = 0.5) + xlim(0,12)+ theme_bw()
kruskal.test(campaign  ~ y, data = bank)
# p = 1.565e-05 < 0.05

ggplot(data = bank, aes(x = pdays , fill = y)) + geom_density(alpha = 0.5) + xlim(0,750)+ theme_bw()
kruskal.test(pdays  ~ y, data = bank)
# p = 2.2e-16 < 0.05

ggplot(data = bank, aes(x = previous , fill = y)) + geom_density(alpha = 0.5) + xlim(-1,7)+ theme_bw()
kruskal.test(previous  ~ y, data = bank)
# p = 2.2e-16 < 0.05

str(bank)
summary(bank)

# Nakon detaljne analize i sredjivanja podataka, uklonili smo sve nerelevantne podatke i dataset pripremili za 
#  kreiranje modela. Dataset je i dalje neuravnotezen i na to mora da se obrati paznja u daljoj analizi.

# Sredjene podatake cuvamo u RData formatu 
write.csv(bank,"data/bank_cleaned_data.csv")
saveRDS(bank, "data/cleaned_bank_data.RData")
