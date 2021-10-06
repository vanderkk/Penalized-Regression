library(glmnet)
library(MASS)
library(caret)

setwd("C:/Users/kayla/Documents/STAT 6801 - Statistical Learning/Project")

cancer <- read.table(file = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/breast-cancer.data"
             , sep = ","
             , header = FALSE)
names(cancer) <- c('class', 'age', 'menopause', 'tumorsize', 'inv.nodes', 'node.caps', 'deg.malig', 'breast', 'breast.quad', 'irradant')

# Remove observations with unknown values
cancer <- cancer[!(cancer$node.caps == "?" | cancer$breast.quad == "?"),]

# remove age group 20-29, and inv nodes 24-26 since there is only one observation of each category.
cancer <- cancer[!(cancer$age == "20-29" | cancer$inv.nodes =="24-26"),]

# Make sure data is in the correct form
cancer$age <- droplevels(cancer$age)
cancer$inv.nodes <- droplevels(cancer$inv.nodes)
cancer$menopause <- ifelse(cancer$menopause == "lt40" | cancer$menopause == "ge40", "postmeno", "premeno")
cancer$menopause <- as.factor(cancer$menopause)
cancer$deg.malig <- as.factor(cancer$deg.malig)
cancer$node.caps <- factor(cancer$node.caps)
cancer$breast.quad <- factor(cancer$breast.quad)
cancer$class <- ifelse(cancer$class == "recurrence-events", 1, 0)
cancer$class <- as.factor(cancer$class)

summary(cancer)

#### Testing and training sets ####
set.seed(1101)
trainDataIndex <- createDataPartition(cancer$class, p=0.7, list = F)  # 70% training data
train.data <- cancer[trainDataIndex, ]
test.data <- cancer[-trainDataIndex, ]

#Up Sampling due to imbalance in recurrance category
up.train <- upSample(x = train.data[, -1],
                     y = train.data$class)
table(up.train$Class)


#### Classical Model #####
set.seed(1101)
model.classical <- glm(Class ~ ., data=up.train, family=binomial(link = "logit"))
summary(model.classical)

classic.coef <- as.data.frame(as.matrix(coef(model.classical)))
colnames(classic.coef) <- "Classic Model"
classic.coef$variable <- row.names(classic.coef)
row.names(classic.coef) <- NULL

#### Cross-validation ####
# remove the absorption of the base factor for model.matrix
contr.Dummy <- function(contrasts, ...){
  conT <- contr.treatment(contrasts=FALSE, ...)
  conT
}
options(contrasts=c(ordered='contr.Dummy', unordered='contr.Dummy'))

cancer.mat <- model.matrix( ~ .-1, up.train[,-10])
cancerclass <- up.train$Class
cancerclass2 <- ifelse(up.train$Class == 1, 1, 0)

#### Penalized regression models ####
# Ridge
set.seed(10403)
model.selection.ridge <- cv.glmnet(x= cancer.mat, y=cancerclass, nfolds = 5, family="binomial", alpha=0) #alpha = 0 ==> Ridge

lambda.ridge <- model.selection.ridge$lambda.min
lambda.ridge2 <- model.selection.ridge$lambda.1se

plot(model.selection.ridge, cex.lab = 1.5)
abline(v = log(lambda.ridge2), lty = 3, col="red")
legend("bottomright", lty = c(3,3), col = c("black", "red"), legend = c("Min Deviance", "1 Std err"), bty="n")
dev.copy(png, 'Ridge_crossvalidation.png')
dev.off()

# LASSO
set.seed(54149355)
model.selection.lasso <- cv.glmnet(x= cancer.mat, y=cancerclass, nfolds = 5, family="binomial", alpha=1) #alpha = 1 ==> Lasso

lambda.lasso <- model.selection.lasso$lambda.min
lambda.lasso2 <- model.selection.lasso$lambda.1se

plot(model.selection.lasso, cex.lab = 1.5)
abline(v = log(lambda.lasso2), lty = 3, col="red")
legend("bottomleft", lty = c(3,3), col = c("black", "red"), legend = c("Min Deviance", "1 Std err"), bty="n")
dev.copy(png, 'Lasso_crossvalidation.png')
dev.off()

#### Final penalized models ####
model.ridge <- glmnet(x= cancer.mat, y=cancerclass, family="binomial", alpha=0, lambda = lambda.ridge)
model.ridge2 <- glmnet(x= cancer.mat, y=cancerclass, family="binomial", alpha=0, lambda = lambda.ridge2)

ridge2.coef <- coef(model.ridge2)
ridge.coef <- coef(model.ridge)

plot(glmnet(x= cancer.mat, y=cancerclass, family="binomial", alpha=0)
     , xvar="lambda", label=TRUE, cex.lab = 1.5)
abline(v = log(lambda.ridge), lty = 3)
abline(v = log(lambda.ridge2), lty = 3, col="red")
legend("bottomright", lty = c(3,3), col = c("black", "red"), legend = c("Min Deviance", "1 Std err"), bty="n")
dev.copy(png, 'Ridge_coeffients_shirnkage_withlambda.png')
dev.off()

model.lasso <- glmnet(x= cancer.mat, y=cancerclass, family="binomial", alpha=1, lambda = lambda.lasso)
model.lasso2 <- glmnet(x= cancer.mat, y=cancerclass, family="binomial", alpha=1, lambda = lambda.lasso2)

lasso.coef <- coef(model.lasso)
lasso2.coef <- coef(model.lasso2)

plot(glmnet(x= cancer.mat, y=cancerclass, family="binomial", alpha=1)
     , xvar="lambda", label=TRUE, cex.lab = 1.5)
abline(v = log(lambda.lasso), lty = 3)
abline(v = log(lambda.lasso2), lty = 3, col="red")
legend("bottomright", lty = c(3,3), col = c("black", "red"), legend = c("Min Deviance", "1 Std err"), bty="n")
dev.copy(png, 'Lasso_coeffients_shirnkage_withlambda.png')
dev.off()

#### Coeficient Table ####
coef.tab <- as.data.frame(cbind(as.matrix(ridge.coef)
                  , as.matrix(ridge2.coef)
                  , as.matrix(lasso.coef)
                  , as.matrix(lasso2.coef)))
colnames(coef.tab) <- c("Ridge Model", "Ridge 2", "LASSO", "LASSO 2")
coef.tab$variable <- row.names(coef.tab)
row.names(coef.tab) <- NULL

coef.tab <- merge(coef.tab, classic.coef
                   , by="variable"
                   , all=TRUE)
cbind(coef.tab[,1], round(coef.tab[,(2:6)],3))


##### Predictions ####
test.matrix <- model.matrix( ~ .-1, test.data[,-1])
test.actual <- test.data$class

pred.fun <- function(testset, actual, model, lambda.corr = TRUE, type){
  
  if(lambda.corr == TRUE){
    predicted <- predict(model, newx=testset, type="response")

  } else {
    predicted <- predict(model, newdata=testset[,-1], type="response")
}
  
  binary <- ifelse(predicted > 0.5, 1, 0)
  binary <- factor(binary)
  accuracy <- mean(binary == actual)
  fac0.actual <- actual[actual == 0]
  fac1.actual <- actual[actual == 1]
  t1err <- sum(binary[actual == 0] != fac0.actual)/length(fac0.actual)
  t2err <- sum(binary[actual == 1] != fac1.actual)/length(fac1.actual)
  toterr <- mean(binary != actual)
  
  return(list("t1err" = t1err, "t2err" = t2err, "toterr" = toterr))
}

pred.classical <- pred.fun(testset = test.data, actual = test.actual, model = model.classical, lambda.corr = FALSE)

pred.ridge <- pred.fun(testset = test.matrix, actual = test.actual, model = model.ridge, lambda.corr = TRUE)
pred.lasso <- pred.fun(testset = test.matrix, actual = test.actual, model = model.lasso, lambda.corr = TRUE)

pred.ridge2 <- pred.fun(testset = test.matrix, actual = test.actual, model = model.ridge2, lambda.corr = TRUE)
pred.lasso2 <- pred.fun(testset = test.matrix, actual = test.actual, model = model.lasso2, lambda.corr = TRUE)

pred.summary <- rbind(pred.classical
                      , pred.ridge
                      , pred.ridge2
                      , pred.lasso
                      , pred.lasso2)
pred.summary