# This script is used in order to evaluate the ensemble. 
# The evaluation includes the comparison of the ensemble with the best performing model from the library using performance profile plots and st
# tistical hypothesis testing.

# --- set up ---
rm(list=ls())
require(ggplot2)
require(reshape2)
setwd("thesis_experiments")


# --- load data---
# get performance of ensemble
ensemble_path <- "system_evaluation/ensemble_results/ensemble"
directories   <- list.dirs(path=ensemble_path,recursive = FALSE,full.names = TRUE)
means_ensemble <- c()
for(i in seq(1,length(directories))) {
  info <- paste(directories[[i]], "ensemble_accuracies.RData",sep="/" )
  load(info)
  means_ensemble <- c(means_ensemble,mean(unlist(accuracies)))
}

# get performance of best performing models
ensemble_path <- "system_evaluation/ensemble_results/best_model"
directories_2   <- list.dirs(path=ensemble_path ,recursive = FALSE,full.names = TRUE)
means_best <- c()
for(i in seq(1,length(directories_2))) {
  info <- paste(directories_2[[i]], "ensemble_accuracies.RData",sep="/" )
  load(info)
  means_best <- c(means_best,mean(unlist(accuracies)))
}
total_accuracies <- data.frame(ensemble = means_ensemble, bestModel = means_best)

# --- draw performance plot --
performances <- total_accuracies
perform_profiles <- data.frame()
# find minimum of each column
alg_min <- apply(performances, 1, function(x) min(x, na.rm = TRUE))
# compute ratio of each element by dividing with smallest dataset performance
perform_profiles <- performances/alg_min
# replace all NaN's with twice the max ratio (a Nan represents an unsolved dataset)
max_ratio <- max(perform_profiles)
perform_profiles[is.na(perform_profiles)] <- 2*max_ratio
# sort each column in ascengind order
perform_profiles <- as.data.frame(apply(perform_profiles, 2, sort))
# -------- plot stair graph (one line per algorithm, showing what percentage of algorithms performs better on each dataset)--------
# find points to plot
perform_profiles$x <- (seq(1, nrow(perform_profiles)))/nrow(perform_profiles)

perform_profiles <- melt(perform_profiles ,  id.vars = 'x', variable.name = 'algorithm')
ggplot(perform_profiles, aes(value, x)) + geom_line(aes(linetype =algorithm)) +
  ggtitle('') + 
  labs(x="t",y="P")  + geom_point(aes(shape = algorithm))

# --- perform hypothesis testing ---
wilcox.test( means_ensemble, means_best, paired = TRUE)
