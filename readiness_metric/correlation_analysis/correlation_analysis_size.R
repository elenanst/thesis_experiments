# This script performs correlation analysis on our anticipation metric and the performance our our system.
#
# Correlation analysis consists in visualizations and computation of the correlation coefficient.
# Correlation is examined is 3 cases:
# a. correlation between anticipation metric and system's performance
# b. correlation between anticipation metric and difference of system's performance to benchmark method
# c. correlation between outlier/non-outlier labels and difference of system's performance to benchmark method

# --- set up ---
rm(list=ls())
setwd("automl_experiments")

RMSE <- function(x,y) {
  sqrt( sum( (x - y)^2 , na.rm = TRUE ) / length(x) )
}

# --- load data ---
# HPP model and data
load("../automl/workspace/HPP/models/AnnClassifier/size/model.RData")
hpp_data  <- read.csv("../automl/workspace/HPP/models/AnnClassifier/size/parameters.csv",
                      header = TRUE, sep=",", stringsAsFactors=FALSE)

# training metafeatures
metafeatures   <- read.csv("dataset_outliers/training_metafeatures_unprocessed.csv",
                           header = TRUE, sep=",", stringsAsFactors=FALSE)
metafeatures   <- metafeatures[,colnames(metafeatures) %in% hpp_data$metafeatures]
metafeatures$X <- NULL
# remove inappropriate values
inap_remover = new('InapRemover')
metafeatures <- inap_remover$removeInfinites(metafeatures,
                                             inf_action = list( act= "replace", rep_pos = 1, rep_neg = 0))
metafeatures <- inap_remover$removeUnknown(metafeatures)
# remove constant columns
metafeatures <- metafeatures[,sapply(metafeatures, function(v) var(v, na.rm=TRUE)!=0)]
temp <- scale(metafeatures)
means          <- attr(temp, "scaled:center")
scales         <- attr(temp, "scaled:scale")
metafeatures   <- as.data.frame(scale(metafeatures))


# testing metafeatures
metafeatures_test   <- read.csv("dataset_outliers/testing_metafeatures.csv",
                                header = TRUE, sep=",", stringsAsFactors=FALSE)
# remove inappropriate values
inap_remover = new('InapRemover')
metafeatures <- inap_remover$removeInfinites(metafeatures,
                                             inf_action = list( act= "replace", rep_pos = 1, rep_neg = 0))
metafeatures <- inap_remover$removeUnknown(metafeatures)
# remove constant columns
metafeatures <- metafeatures[,sapply(metafeatures, function(v) var(v, na.rm=TRUE)!=0)]




# class of hp
test_class <- read.csv("data/opt/nnet_opt.csv",
                       header = TRUE, sep=",", stringsAsFactors=FALSE)
names      <- lapply(test_class$X, function(x) substr(x, 1, nchar(x)-8))
names      <- paste(names, ".csv", sep="")
test_class <- test_class[names %in% metafeatures_test$X,]
test_class <- test_class$size

# clean test metafeatures
metafeatures_test[is.na(metafeatures_test)] <- 0
metafeatures_test   <- metafeatures_test[, colnames(metafeatures_test) %in% hpp_data$metafeatures]
metafeatures_test   <- as.data.frame(scale(metafeatures_test, center = means, scale = scales))
metafeatures_test$X <- NULL

optimal_k <- floor(sqrt(nrow(metafeatures)))

# --- compute training distance scores ---

distance_scores <- c()
for(i in seq(1, nrow(metafeatures))) {
  # calculate distance from all examples
  distance <- apply(metafeatures, 1, function(x) {sqrt(sum((x - metafeatures[i,]) ^ 2))})
  # find k-nearest examples
  distance <- distance[order(distance)[1:optimal_k]]
  # find average distance
  distance <- mean(distance)
  distance_scores <- c(distance_scores, distance)
}
write.csv(data.frame(distances = distance_scores), "distances_size.csv")
dataset <- data.frame(X= distance_scores)
library(tikzDevice)
tikz('../automl_experiments/distances_hist_size.tex', standAlone = TRUE, width=7, height=5)
ggplot(dataset, aes(x = X)) + geom_histogram(aes(y = ..density..)) + geom_density(adjust = 1.5,kernel = "gaussian")
dev.off()
density_estimate <- density(distance_scores,adjust=1.5,kernel = "gaussian")
first_q          <- 2.0905 
third_q          <- 6.5968
inter_range      <- third_q - first_q 
low_bound        <- first_q    - 1.5 *inter_range
upper_bound      <-  third_q+ 1.5 *inter_range

# --- compute testing distance scores ---
distance_scores_test <- c()
optimal_hp      <- c()
error <- c()
metafeatures_test <- metafeatures_test[,colnames(metafeatures_test) %in% colnames(metafeatures)]
for(i in seq(1,nrow(metafeatures_test))) {
  distance <- apply(metafeatures, 1, function(x) {((sum((x - metafeatures_test[i,]) ^ 2)))})
  distance <- na.omit(distance)
  # find k-nearest examples
  distance <- distance[order(distance)[1:optimal_k]]
  # find average distance
  distance <- mean(distance)
  distance_scores_test <- c(distance_scores_test , distance)
  # get prediction of k 
  optimal_hp <- c(optimal_hp, (10^predict(model, metafeatures_test[i,])))
  error     <- c(error, RMSE(round(10^predict(model, metafeatures_test[i,])), test_class[i]))
}

plot(distance_scores_test, error)
coefficient  <- cor(distance_scores_test,error)

inter_range <- third_q - first_q
upper_bound <- third_q + 0.3*inter_range 
indexes    <- (distance_scores_test> upper_bound)
# box-plot
df  <- data.frame(group = indexes, y = error)
ggplot(df, aes(x = group, y = error)) +
  geom_boxplot(fill = "grey80", colour = "blue") +
  scale_x_discrete() + xlab("Treatment Group") +
  ylab("Dried weight of plants")
