### Extra Credit Group Project
### Solar Flares

library(dplyr)
library(nnet)
library(caret)
library(ggplot2)
library(tree)
library(gbm)

setwd("C:/Users/kphas_000/Desktop/PREDICT_422/ExtraCredit/")

### Useful functions

se <- function(x) sd(x)/sqrt(length(x))

### Import data
flare2 <- read.csv("flare.data2", skip = 1, header = FALSE, sep = " ")
flare_names <- c("zurich_class", "spot_size", "spot_distrib", "activity",
                 "evolution", "act_code", "hist_complex", "become_complex",
                 "area", "area_largest", "c_class", "m_class", "x_class")
colnames(flare2) <- flare_names
summary(flare2)
str(flare2)

# Make columns factors
flare2[, 4:9] <- lapply(flare2[, 4:9], FUN = as.factor)
str(flare2)

# Histograms of target variables
hist(flare2$c_class)
hist(flare2$m_class)
hist(flare2$x_class)

# Frequencies of target variables
table(flare2$c_class)
table(flare2$m_class)
table(flare2$x_class)


### Create test and training sets
n <- dim(flare2)[1]
set.seed(1)
test <- sample(n, round(n/4)) # randomly sample 25% test
data.train <- flare2[-test, ]
data.test <- flare2[test, ]

# Create model matrices
x       <- model.matrix(c_class ~ ., data = flare2)[,-1] # 1st col is 1's
x.train <- x[-test,]          # define training predictor matrix
x.test  <- x[test,]           # define test predictor matrix
y       <- flare2$c_class     # define response variable
y.train <- y[-test]           # define training response variable
y.test  <- y[test]            # define test response variable
n.train <- dim(data.train)[1] # training sample size = 800
n.test  <- dim(data.test)[1]  # test sample size = 266



### Artificial neural network model
nnet.fit <- nnet(c_class ~ . - m_class - x_class, data = data.train, size = 2) # fit without m_class and x_class columns
nnet.predict <- predict(nnet.fit, data.test)
ann.mse <- mean((y.test - nnet.predict)^2)
ann.mse # [1] 0.7180451128
ann.se <- se((y.test - nnet.predict)^2)
ann.se # [1] 0.1573765894
#plot(data.test$c_class, nnet.predict, main = "Artificial Neural Network Predictions vs Actual",
#     xlab = "Actual", ylab = "Predictions", pch = 19, col = "blue")



### Tree model
tree.flare <- tree(c_class ~ . - m_class - x_class, data = data.train)
summary(tree.flare) # zurich_class, spot_size, spot_distrib, and activity are included in tree
plot(tree.flare)
text(tree.flare, pretty = 0)
tree.flare
yhat.tree <- predict(tree.flare, data.test)
tree.mse <- mean((y.test - yhat.tree)^2)
tree.mse # [1] 0.5562123343
tree.se <- se((y.test - nnet.predict)^2)
tree.se # [1] 0.1573765894

# prune the tree to see if we get better results
set.seed(1)
cv.flare <- cv.tree(tree.flare)
plot(cv.flare$size, cv.flare$dev, type = "b")
prune.flare <- prune.tree(tree.flare, best = 2)
plot(prune.flare)
text(prune.flare, pretty = 0)
yhat.tree <- predict(prune.flare, data.test)
tree.mse <- mean((y.test - yhat.tree)^2)
tree.mse # [1] 0.5724697791
tree.se <- se((y.test - nnet.predict)^2)
tree.se # [1] 0.1573765894
# unfortunately the model did not improve when pruning



### Boosting model
set.seed(1)
boost.flare <- gbm(c_class ~ . - m_class - x_class, data = data.train, n.trees = 5000, interaction.depth = 4)
summary(boost.flare)
par(mfrow=c(1,2))
plot(boost.flare, i = "zurich_class")
plot(boost.flare, i = "spot_size")
yhat.boost <- predict(boost.flare, newdata = data.test, n.trees = 5000)
boost.mse <- mean((y.test - yhat.boost)^2) 
boost.mse # [1] 0.5634364807
boost.se <- se((y.test - yhat.boost)^2)
boost.se # [1] 0.1202052845

## fit using only zurich_class and spot_size since they are the most important variables based on summary
boost.flare <- gbm(c_class ~ zurich_class + spot_size, data = data.train, n.trees = 5000, interaction.depth = 4)
yhat.boost <- predict(boost.flare, newdata = data.test, n.trees = 5000)
boost.mse <- mean((y.test - yhat.boost)^2) 
boost.mse # [1] 0.5486130138
boost.se <- se((y.test - yhat.boost)^2)
boost.se # [1] 0.1162284558
# slight improvement, but not much



# Model Evaluations
results.mse <- c(ann.mse, tree.mse, boost.mse)
results.se <- c(ann.se, tree.se, boost.se)
results.model <- c("ANN", "Tree", "Boosting")
results <- data.frame(results.model, results.mse, results.se)
colnames(results) <- c("Model", "MSE", "SE")
results <- arrange(results, MSE)
results # %>% write.csv("modresults.csv", row.names = F)

ggplot(results, aes(x = reorder(Model, desc(MSE)), y = MSE)) +
    geom_bar(position = position_dodge(), stat = "identity", width = .8, 
             fill = "lightblue") +
    geom_errorbar(aes(ymin = MSE - SE, ymax = MSE + SE), width = .4) +
    geom_label(label = round(results$MSE, 3)) +
    labs(x = "Model")
# ggsave(filename = "msebarplot.png", width = 7, height = 4)