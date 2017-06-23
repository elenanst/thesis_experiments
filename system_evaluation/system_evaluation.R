# This scripts performs the evaluation of the ADS system.
# The evaluation consists in comparison of ADS' results with other baseline methods through statistical tests and performance plots.
# Baseline methods include grid search and bayesian optimization.


# --- set up ---
setwd("thesis_experiments")
require(ggplot2)
require(reshape2)
require(PMCMR)

# --- load data ---
# load grid search data
grid_path <- "baseline_methods/models/simple"
directories <- list.dirs(path=grid_path,recursive = FALSE,full.names = TRUE)
grid_accuracies <- list()
for (i in 1:length(directories)) {
  load(file.path(directories[[i]],"grid_knn.RData"))
  knn_acc <- model$experiment_accuracies
  load(file.path(directories[[i]],"grid_nnet.RData"))
  nnet_acc <- model$experiment_accuracies
  load(file.path(directories[[i]],"grid_tree.RData"))
  tree_acc <- model$experiment_accuracies
  load(file.path(directories[[i]],"enseble_accuracies.RData"))
  ensemble_acc <- accuracies
  grid_accuracies[[directories[[i]]]] <- list(knn=knn_acc, nnet=nnet_acc, tree = tree_acc, ensemble=ensemble_acc)
}

# load bayesian optimization data 
tpe_accuracies <- as.data.frame(matrix(nrow=22,ncol=3 ))
tpe_path       <- "data/optimization"
knn_tpe        <- read.csv(paste(tpe_path, "bayesian/knn_opt.csv", sep="/"),
                           header = TRUE, sep=",", stringsAsFactors=FALSE)
# remove failed optimizations
knn_ind       <- c(16, 17, 18,23,25,26,33,34,45,48,49,55,
                   71,77,82,85,91,103)
knnTpe        <- 1-knn_tpe[knn_ind,4]
files         <- knn_tpe$X
files         <- files[knn_ind]
nnet_tpe      <- read.csv(paste(tpe_path, "bayesian/nnet_opt.csv", sep="/"),
                    header = TRUE, sep=",", stringsAsFactors=FALSE)
# remove failed optimizations
nnet_ind <- c(15, 16,17, 22,24,25,32,33,44,47,48,54,
             70,76,81,84,89,101)
nnetTpe <- 1-nnet_tpe[nnet_ind,5]
tree_tpe <- read.csv(paste(tpe_path, "bayesian/tree_opt.csv", sep="/"),
                    header = TRUE, sep=",", stringsAsFactors=FALSE)
# remove failed optimizations
tree_ind <- c(16, 17, 18,23,25,26,33,34,45,48,49,55,
             71,77,82,85,91,103)
treeTpe <- 1-tree_tpe[tree_ind,4]

ensemble_path <- "baseline_methods/tpe_models"
ensembleTpe <- c()
directories <- list.dirs(path=ensemble_path,recursive = FALSE,full.names = TRUE)
directories <- directories[-c(7,15,19,21)]
files       <- list.dirs(path=ensemble_path,recursive = FALSE,full.names = FALSE)
files       <- files[-c(7,15,19,21)]
for (i in 1:length(directories)) {
 ensemble_file <- paste(directories[[i]], "model/model_files/enseble_accuracies.RData",sep="/")
 load(ensemble_file)
 ensembleTpe[i] <- mean(unlist(accuracies))
}
tpe_accuracies <- cbind(knnTpe, nnetTpe, treeTpe,ensembleTpe)

# load ADS results
results_path <- "../automl/workspace/results"
directories  <- list.dirs(path=results_path,recursive = FALSE,full.names = TRUE)
automl       <- c()
for(i in seq(1,length(directories))) {
  results <- paste(directories[[i]], "experiment_info.Rdata", sep ="/")
  load(results)
  automl[i] <- mean(unlist(data$ensemble$performance))
}

# --- draw performance plot ---
# convert performances to desired format
total_accuracies <-grid_accuracies
accuracies <- as.data.frame(matrix(nrow = length(total_accuracies),ncol=4))
for (dataset in 1:length(total_accuracies)) {
  accuracies[dataset,1] <- mean(unlist(total_accuracies[[dataset]][["knn"]]))

  accuracies[dataset,2] <- mean(unlist(total_accuracies[[dataset]][["nnet"]]))
  accuracies[dataset,3] <- mean(unlist(total_accuracies[[dataset]][["tree"]]))
  accuracies[dataset,4] <- mean(unlist(total_accuracies[[dataset]][["ensemble"]]))
}
colnames(accuracies)[1] <- "knnGrid"
colnames(accuracies)[2] <- "nnetGrid"
colnames(accuracies)[3] <- "treeGrid"
colnames(accuracies)[4] <- "ensembleGrid"
automl_indexes <- c(7, 15, 19,21)
accuracies <- accuracies[-automl_indexes,]
tpe_accuracies <- tpe_accuracies[automl_indexes,]
total_accuracies <- cbind(accuracies,automl)
total_accuracies <- cbind(tpe_accuracies,automl)
rownames(total_accuracies) <- files

# draw plot
setwd("../automl")
source("build_script.R")
hypothesis_tester <- HypothesisTester$new()
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
# plot
perform_profiles <- melt(perform_profiles ,  id.vars = 'x', variable.name = 'algorithm')
ggplot(perform_profiles, aes(value, x)) + geom_line(aes(linetype =algorithm)) +
  ggtitle('Performance profile plot') + geom_point(aes(shape = algorithm)) + 
  labs(x="t",y="P") 


# ---perform statistical testing ---
hypothesis_tester$rankTest(results = performances, methods = c("friedman"))
posthoc.friedman.nemenyi.test(formula = Performance ~ Method | Dataset , data = Performances)
