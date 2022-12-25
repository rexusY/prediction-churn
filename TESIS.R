library(plyr)
library(corrplot)
library(ggplot2)
library(gridExtra)
install.packages("ggthemes")
install.packages("randomForest")
install.packages("party")
install.packages("ctree")
install.packages("ROCR")
install.packages("pROC")
install.packages("dummies")
install.packages("pastecs")
install.packages("DMwR")
library(ggthemes)
library(ctree)
library(caret)
library(MASS)
library(randomForest)
library(party)
library(tidyverse)
library(miscset)
library(pastecs)
install.packages("modeest")
require(modeest)
library(e1071)
library(caret)
library(devtools)

library(dplyr)
library(rpart)
library(randomForest)
library(ROCR)
library(rpart.plot)
library(dummies)
library(caret)
library(ggplot2)
library(pROC)
library(DT)
install.packages("earth")
install.packages("vip")
install.packages("pdp")
install.packages("mda")
install.packages("DMwR")
# Helper packages
library(dplyr)     # for data wrangling
library(ggplot2) # for awesome plotting
library(mda) 
library(DMwR)
# Modeling packages
library(earth)     # for fitting MARS models
library(caret)     # for automating the tuning process

# Model interpretability packages
library(vip)       # for variable importance
library(pdp)       # for variable relationships


rm(list=ls(all=T))
options(digits=4, scipen=12)
library(dplyr)
library(rpart)
library(rpart.plot)
library(caret)
library(randomForest)
library(caTools)
source('DPP.R')

install.packages("zoom")
library(zoom)
library(ISLR)

#import data set
#dataset<-read.table("./CHURN TELKOM BARU 2.csv",header = T,sep = ";")
#dataset<-read.table("./CHURN TELKOM COBA.csv",header = T,sep = ";")
dataset<-read.table("./DATA CHURN NEW.csv",header = T,sep = ";")

str(dataset)



#show to statistic descriptive
res <- stat.desc(dataset[,10])
round(res, 2)


#convert variable to number
dataset$USAGE <- as.numeric(dataset$USAGE)

plot(dataset$USAGE, dataset$GANGGUAN)
plot(dataset$USAGE, dataset$GANGGUAN, xlab="USAGE", ylab="GANGGUAN")

pairs(dataset[c("USAGE","GANGGUAN")], 
      main="Matrix plot", pch=22, 
      bg=c("red", "yellow")[unclass(dataset$CHURN)])
library(zoom) # Invoke the Library
# Call plot
zm()
install.packages("TeachingDemos")
library(TeachingDemos)
zoomplot( locator(2) )
zoomplot.zoom(fact=2,x=0,y=0)
#YELLOW is Churn, RED is active


#density plot GANGGUAN

avgGANGGUAN.nodef = mean(dataset[dataset$CHURN == "CHURN", "GANGGUAN"])
avgGANGGUAN.nodef
avgGANGGUAN.def   = mean(dataset[dataset$CHURN == "AKTIF", "GANGGUAN"])
avgGANGGUAN.def


ggplot(dataset, aes(GANGGUAN, fill=CHURN)) + 
  geom_density(alpha=.5) + 
  geom_vline(data=dataset, 
             mapping=aes(xintercept=avgGANGGUAN.nodef), color="red") +
  geom_vline(data=dataset, 
             mapping=aes(xintercept=avgGANGGUAN.def), color="dark green")




#plot DIVISI
variables <- list('CHURN')
plotG <- list() #x

for (i in variables){
  plotG <-  ggplot(dataset, aes_string(x = i, fill = as.factor(dataset$CHURN)))+
    geom_bar( position = "stack")+ scale_fill_discrete(name = "churn")+
    scale_x_discrete(labels = function(x) str_wrap(x, width = 10))
  
  print(plotG)
}



#Show Correlation plot
dataset$NOMOR <- NULL
dataset$DETAIL.CHURN <- NULL
dataset$INET <- NULL
numeric.var <- sapply(dataset, is.numeric)
corr.matrix <- cor(dataset[,numeric.var])
glimpse(dataset)
corrplot(corr.matrix, main="\n\nCorrelation Plot for Numerical Variables", method="number")
#remove SPEED, LAMA PEMAKAIAN because variable is corelation
BJKB
#Partition Data to be data training and data testing
intrain<- createDataPartition(dataset$CHURN,p=0.7,list=FALSE)
set.seed(2017)
training<- dataset[intrain,]
testing<- dataset[-intrain,]
glimpse(training)
dim(training); dim(testing)

library(DMwR)
training$CHURN <- as.factor(training$CHURN)
trainingSMOTE2 <- SMOTE(CHURN ~ ., training, perc.over = 700, perc.under=122)
trainingSMOTE <- SMOTE(CHURN ~ ., training, perc.over = 100, perc.under=200)
summary(trainingSMOTE$CHURN)
prop.table(table(trainingSMOTE$CHURN))
#training$CHURN <- as.numeric(training$CHURN)



#----------------------------------#
#   Modeling data Regresi Logistic#
#----------------------------------#

LogModel <- glm(CHURN~ TELP+TV+TECH+USAGE+BILL+GANGGUAN+CLASS.PELANGGAN+INDIHOME.DESC+P+SPEED+LAMA.PEMAKAIAN,family=binomial,data=trainingSMOTE)


#LogModel <- glm(CHURN~ TELP+TV+TECH+USAGE+BILL+GANGGUAN+CLASS.PELANGGAN+INDIHOME.DESC+P+SPEED+LAMA.PEMAKAIAN,family=binomial,data=training)

print(summary(LogModel))
coef(LogModel)

predictions_LogModel <- predict(LogModel, testing, type = "response")

Churn_LogModel <- ifelse(predictions_LogModel >= c[[1]], 
                     "CHURN", "AKTIF")

Churn_LogModel <- as.factor(Churn_LogModel)
testing$CHURN<- as.factor(testing$CHURN)
cm_log <- confusionMatrix(testing$CHURN, Churn_LogModel)
cm_log

#Stepwise Model
step.model <- LogModel %>% stepAIC(trace = FALSE)
print(summary(step.model))
coef(step.model)
step.model$anova

predictions_step <- predict(step.model, testing, type = "response")
pred_step <- prediction(predictions_step, testing$CHURN)
#Make Graphs ROC
plot(performance(pred_step, "tpr", "fpr"), colorize = TRUE)

auc <- performance(pred_step, measure = "auc")
auc <- auc@y.values[[1]]
auc

#Sensitivity, Threshold, Specifity
roc_step <- roc(response = testing$CHURN, predictor = predictions_step)
c <- coords(roc_step, "best", "threshold")
c

Churn_step <- ifelse(predictions_step >= c[[1]], 
                     "CHURN", "AKTIF")
Churn_step<- as.factor(Churn_step)
testing$CHURN<- as.factor(testing$CHURN)
cm_log <- confusionMatrix(testing$CHURN, Churn_step)
cm_log
auc




#stepwise model training
# Make prediction
#probabilitiesStepTrain <- predict(step.model, training, type = "response")
#head(probabilitiesStepTrain)
#predicted.classesStepTrain <- ifelse(probabilitiesStepTrain > 0.5, "CABUT", "AKTIF")
# Prediction accuracy
#observed.classes <- training$CHURN
#mean(predicted.classesStepTrain == observed.classes)

#training$CHURN <- as.character(training$CHURN)
#training$CHURN [training$CHURN =="No"] <- "0"
#training$CHURN [training$CHURN =="Yes"] <- "1"
#fitted.results <- predict(step.model,newdata=training,type='response')
##fitted.results <- ifelse(fitted.results > 0.3,"CABUT", "AKTIF")
#misClasificError <- mean(fitted.results != training$CHURN)
#print(paste('Logistic Regression Accuracy',1-misClasificError))
#print("Confusion Matrix for Logistic Regression"); table(training$CHURN, fitted.results > 0.5)



##stepwise model testing
# Make predictions
#probabilitiesStepTest <- predict(step.model, testing, type = "response")
#head(probabilitiesStepTest)
#predicted.classesStepTest <- ifelse(probabilitiesStepTest > 0.5, "CABUT", "AKTIF")
## Prediction accuracy
#observed.classes <- testing$CHURN
#mean(predicted.classesStepTest == observed.classes)


#testing$CHURN <- as.character(testing$CHURN)
#testing$CHURN [testing$CHURN =="No"] <- "0"
#testing$CHURN [testing$CHURN =="Yes"] <- "1"
#fitted.results <- predict(step.model,newdata=testing,type='response')
#misClasificError <- mean(fitted.results != testing$CHURN)
#print(paste('Logistic Regression Accuracy',1-misClasificError))
#print("Confusion Matrix for Logistic Regression"); table(testing$CHURN, fitted.results > 0.5)


#----------------------------------#
#   Modeling data CART             #
#----------------------------------#
rp <- rpart(CHURN~ DIVISI+CATEGORI+TELP+TV+TECH+USAGE+BILL+GANGGUAN+CLASS.PELANGGAN+INDIHOME.DESC+P+SPEED+LAMA.PEMAKAIAN, data = trainingSMOTE, method = "class")
predictions_rp <- predict(rp, testing, type = "class")
head(predictions_rp)

rpart.plot(rp, type=1, extra=100, branch.lty=3, box.palette="RdYlGn", tweak = 1.2, fallen.leaves = FALSE)
table(predictions_rp, testing$CHURN)
zm()



PredictROCrp = predict(rp, newdata = testing)

predrp = prediction(PredictROCrp[,2], testing$CHURN)
perf = performance(predrp, "tpr", "fpr")
plot(perf, colorize=T)

auc <- performance(predrp, measure = "auc")
auc <- auc@y.values[[1]]
auc



cm_rpart <- confusionMatrix(data=predictions_rp, testing$CHURN)
cm_rpart


#----------------------------------#
#   Modeling data Random Forest   #
#----------------------------------#
tc_rf <- trainControl(method = "repeatedcv",repeats = 2,number = 3, search = "random")
rf_train <- train(CHURN~ DIVISI+CATEGORI+TELP+TV+TECH+USAGE+BILL+GANGGUAN+CLASS.PELANGGAN+INDIHOME.DESC+P+SPEED+LAMA.PEMAKAIAN, data = trainingSMOTE, method = "rf", trainControl = tc_rf) ##Time consuming
plot(varImp(rf_train, scale = F))

PredictROCrf = predict(rf_train,testing,type="prob")
predrf <- prediction(PredictROCrf [,2],testing$CHURN)
perf <- performance(predrf, "tpr", "fpr")
plot(perf, colorize=T)

auc <- performance(predrf, measure = "auc")
auc <- auc@y.values[[1]]
auc




predict_rftrain <- predict(rf_train, testing)
cm_rf <- confusionMatrix(predict_rftrain, testing$CHURN)
cm_rf



#----------------------------------#
#   Modeling data Random SVM   #
#----------------------------------#

svm <- tune.svm(CHURN ~ DIVISI+CATEGORI+TELP+TV+TECH+USAGE+BILL+GANGGUAN+CLASS.PELANGGAN+INDIHOME.DESC+P+SPEED+LAMA.PEMAKAIAN, data = trainingSMOTE, seq(0.5, 0.9, by = 0.1), cost = seq(100, 1000, by = 100), kernel = "radial", tunecontrol = tune.control(cross = 10))

print(svm)
summary(svm)
svm$performances
svmfit <- svm$best.model
pred_svm <- predict(svmfit, testing, type = 'response')


# Plot optimal parameter model's performance on training data
PredictROCsvm = attributes(predict(svmfit, testing, 
                                   decision.values = TRUE))$decision.values
predsvm <- prediction(PredictROCsvm,testing$CHURN)
perf <- performance(predsvm, "tpr", "fpr")
plot(perf, colorize=T)

auc <- performance(predsvm, measure = "auc")
auc <- auc@y.values[[1]]
auc

cm_svm <- confusionMatrix(pred_svm, testing$CHURN)
cm_svm

#----------------------------------#
#   Modeling data Random MARS   #
#----------------------------------#

hyper_grid <- expand.grid(
  degree = 1:1, 
  nprune = seq(2, 40, length.out = 10) %>% floor()
)

#tuned_mars <- train(
#  x = subset(trainingSMOTE2, select = -CHURN),
#  y = trainingSMOTE2$CHURN,
#  method = "earth",
#  trControl = trainControl(method = "cv", number = 10),
#  tuneGrid = hyper_grid
#)

tuned_mars = train( CHURN~DIVISI+CATEGORI+TELP+TV+TECH+USAGE+BILL+GANGGUAN+CLASS.PELANGGAN+INDIHOME.DESC+P+SPEED+LAMA.PEMAKAIAN, trainingSMOTE
                    , method = 'earth'
                    , tuneGrid = hyper_grid
                    , trControl = trainControl( method = 'cv'
                                                       , verboseIter = F
                                                       , savePredictions = T
                                                       , allowParallel = T
                    )
)




tuned_mars$bestTune
pred_mars <- predict(tuned_mars, testing, type = 'prob')
predict_mars<-predict(tuned_mars, testing)
confusion(predict(tuned_mars, testing), testing$CHURN)




PredictROCmars = predict(tuned_mars,testing,type="prob")
predmars <- prediction(PredictROCmars [,2],testing$CHURN)
perf <- performance(predmars, "tpr", "fpr")
plot(perf, colorize=T)

auc <- performance(predmars, measure = "auc")
auc <- auc@y.values[[1]]
auc

cm_mars <- confusionMatrix(predict_mars, testing$CHURN)
cm_mars



# plot results
ggplot(tuned_mars)


#----------------------------------#
# Graph ROC Analysis              #   
#--------------------------------#



# regresi logistic
pred_step <- prediction(predictions_step, testing$CHURN)
perf_step <- performance(pred_step, "tpr", "fpr")
# add=TRUE draws on the existing chart 
plot(perf_step, col=4, main="ROC curves of different machine learning classifier")
# Draw a legend.
legend(0.6, 0.6, c('regresi logistic', 'CART','random forest','svm'), 4:8)

# CART
PredictROCrp = predict(rp, newdata = testing)
predrp = prediction(PredictROCrp[,2], testing$CHURN)
perf_rp <- performance(predrp, "tpr", "fpr")
plot(perf_rp, col=5, add=TRUE)

#Random Forest
PredictROCrf = predict(rf_train,testing,type="prob")
predrf <- prediction(PredictROCrf [,2],testing$CHURN)
perf_rf <- performance(predrf, "tpr", "fpr")
plot(perf_rf, col=6, add=TRUE)

#SVM
PredictROCsvm = attributes(predict(svmfit, testing, 
                                   decision.values = TRUE))$decision.values
predsvm <- prediction(PredictROCsvm,testing$CHURN)
perf_svm <- performance(predsvm, "tpr", "fpr")
plot(perf_svm, col=7, add=TRUE)


#MARS
PredictROCmars = predict(tuned_mars,testing,type="prob")
predmars <- prediction(PredictROCmars [,2],testing$CHURN)
perf_mars <- performance(predmars, "tpr", "fpr")
plot(perf_mars, col=8, add=TRUE)



plot(rnorm(1000),rnorm(1000)) # could be any plot
zm() # navigate the plot


