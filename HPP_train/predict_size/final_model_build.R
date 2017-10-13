# Trains model for predicting size of nnet.

# --- set up ---
rm(list=ls())
setwd("automl")
source("build_script.R")
setwd("../thesis_experiments")
source("HPP_train/math_tools/boot_pi.R")
library(caret)
set.seed(1)
library(tikzDevice)
library(Boruta)
library(stringr)
# --- define functions ---
RMSE <- function(x,y) {
  sqrt( sum( (x - y)^2 , na.rm = TRUE ) / length(x) )
}

remove_outliers <- function(x, na.rm = TRUE, ...) {
  qnt <- quantile(x, probs=c(.25, .75), na.rm = na.rm, ...)
  H <-  10* IQR(x, na.rm = na.rm)
  y <- x
  y[x < (qnt[1] - H)] <- max(x)
  y[x > (qnt[2] + H)] <- min(x)
  y
}

# --- load data ---
training_data                <- read.csv("HPP_train/training_metafeatures.csv",
                                         header = TRUE, sep=",", stringsAsFactors=FALSE)
train_files                  <- training_data$X
training_data$hasNumeric     <- NULL
training_data$hasCategorical <- NULL
training_data$X              <- NULL
class                        <- read.csv("data/optimization/bayesian/nnet_opt.csv",
                                         header = TRUE, sep=",", stringsAsFactors=FALSE)
names      <- lapply(class$X, function(x) substr(x, 1, nchar(x)-8))
names      <- paste(names, ".csv", sep="")
class      <- class[names %in% train_files,]
training_data <- training_data[train_files %in% names, ]
class      <- class$size
testing_data                <- read.csv("HPP_train/testing_metafeatures.csv",
                                        header = TRUE, sep=",", stringsAsFactors=FALSE)
test_files                  <- testing_data$X
testing_data$hasNumeric     <- NULL
testing_data$hasCategorical <- NULL
testing_data$X              <- NULL
test_class                        <- read.csv("data/optimization/bayesian/nnet_opt.csv",
                                              header = TRUE, sep=",", stringsAsFactors=FALSE)
names      <- lapply(test_class$X, function(x) substr(x, 1, nchar(x)-8))
names      <- paste(names, ".csv", sep="")
test_class      <- test_class[names %in% test_files,]
testing_data <- testing_data[test_files %in% names, ]
test_class      <- test_class$size

# --- preprocess metafeatures ---
# --- remove outliers ---
for(i in 1:(ncol(training_data)-1)) {
  training_data[,i] <- remove_outliers(training_data[,i])
}
# --- scale data ---
x             <- scale(training_data)
means         <- attr(x, "scaled:center")
scales        <- attr(x, "scaled:scale")
training_data <- as.data.frame(scale(training_data))
# remove inappropriate values
inap_remover = new('InapRemover')
training_data <- inap_remover$removeInfinites(training_data,
                                              inf_action = list( act= "replace", rep_pos = 1, rep_neg = 0))
training_data <- inap_remover$removeUnknown(training_data)
zero_columns  <- nearZeroVar(training_data, freqCut = 2.5, uniqueCut = 70, saveMetrics =F)
training_data <- training_data[,- zero_columns]
# filtering
correlationMatrix <- cor(training_data[, !(names(training_data) %in% c("Class"))])
highlyCorrelated  <- findCorrelation(correlationMatrix, cutoff=0.75)
training_data     <- training_data[,-highlyCorrelated]
# forward feature selection
keep   <- list()
trials <- 50
for (i in 1:trials) {
  seed <- runif(1,0,100)
  set.seed(seed)
  control <- rfeControl(functions=lmFuncs, method="cv", number=10)
  # run the RFE algorithm
  results <- rfe(training_data, log10( class),
                 sizes=c(1:(ncol(training_data))), rfeControl=control, optsize =5)
  keep[[i]]  <- results$fit
}
counts<- c()
for (i in colnames(training_data))
{
  count <- sapply(keep, function(x) sum(str_count(names(x$coefficients), i)))
  counts <- c(counts,sum(count))
}  
counts
training_data<-as.data.frame(training_data[, counts>(trials/5), drop=FALSE])
training_data$Class <- class

# --- apply preprocessing to testing data
testing_data <- as.data.frame(scale(testing_data, center = means, scale = scales))
testing_data <- as.data.frame(testing_data[, colnames(testing_data) %in% colnames(training_data), drop=FALSE])
testing_data <- inap_remover$removeInfinites(testing_data,
                                             inf_action = list( act= "replace", rep_pos = 1, rep_neg = 0))
testing_data <- inap_remover$removeUnknown(testing_data)

# --- evaluate model ---
fitControl <- trainControl(## 10-fold CV
  method = "repeatedcv",
  number = 10,
  repeats = 1
)

indexes        <- which(class > 10)
trained_model  <- caret::train(log10(Class) ~ ., data = training_data,
                               method = "ranger", 
                               trControl=fitControl)
predictions_with_confidence    <- boot_pi(model = trained_model, pdata = testing_data, n = 80, p = 1)

# --- plot confidence intervals for testing files ---
y                                <- predictions_with_confidence[[1]]
x                                <- seq_along(y)
ci_low                           <- predictions_with_confidence[[2]]
ci_high                          <- predictions_with_confidence[[3]]
indexes <- which(test_class>=10)
tikz('HPP_train/predict_size/intervals.tex', standAlone = TRUE, width=5, height=5)
plot(y, ylim = c(min(c(test_class,ci_low))-0.01, max(c(test_class, ci_high))+0.01), axes = FALSE, ann = FALSE)
axis(1,at=1:23)
axis(2,at=seq(0,max(c(test_class, ci_high))+0.01,0.5))
title(xlab="Dataset Index")
title(ylab="Size")
points(test_class, pch=22)
arrows(x,ci_low,x,ci_high, lty=2, code=3, angle=90, length=0.05)
legend(1,max(c(test_class, ci_high))+0.01, c("prediction","class"), cex=0.8, 
       col=c("black","black"), pch=21:22)
dev.off()
# --- save workspace
save(list = ls(all.names = TRUE), file = "HPP_train/predict_size/workspace.RData", envir = .GlobalEnv)

# --- train and store model and parameters ---
save(trained_model, file = "HPP_train/predict_size/model.RData")
model_p <- list(metafeatures = colnames(training_data), means = means, scales = scales,
                n_boot = 50 , percentage  = 1, enableLog = 1, step = 0.1, count = 0)
save(model_p, file="HPP_train/predict_size/model_parameters.Rdata")