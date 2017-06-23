# This script is used to create benchmark data for models optimized using Tree Parzen Estimator. Benchmark data include the performance of knn, nnet, rpart and forward model ensembleon testing datasets.

## set up
rm(list=ls())
setwd("automl")
source("automl/build_script.R")

## load data
datasets_path <- "workspace/datasets_repo"
files_list = list.files(path = datasets_path,
                        pattern="*.csv", recursive = TRUE)


tpe_path <- "../thesis_experiments/data/optimization/bayesian"
knn_tpe  <- read.csv(paste(tpe_path, "knn_opt.csv", sep="/"),
                    header = TRUE, sep=",", stringsAsFactors=FALSE)
knn_ind  <- c(16, 17, 23,25,26,18,27,33,34,45,48,49,55,
             71,75,77,82,85,86,91,96,103)
knn_tpe  <- knn_tpe[knn_ind,2]
nnet_tpe <- read.csv(paste(tpe_path, "nnet_opt.csv", sep="/"),
                     header = TRUE, sep=",", stringsAsFactors=FALSE)
nnet_ind <- c(15, 16, 22,24,25,17,26,32,33,44,47,48,54,
              70,74,76,81,84,85,89,94,101
)
nnet_tpe <- nnet_tpe[nnet_ind,c(2,3)]
tree_tpe <- read.csv(paste(tpe_path, "tree_alt_opt.csv", sep="/"),
                     header = TRUE, sep=",", stringsAsFactors=FALSE)
tree_ind <- c(16, 17, 23,25,26,18,27,33,34,45,48,49,55,
              71,75,77,82,85,86,91,96,103
)
tree_tpe <- tree_tpe[tree_ind,2]

## train models
for (i in 1:length(files_list)) {
  ### load dataset
  dataset          <- read.csv(paste(datasets_path,files_list[[i]], sep="/"),
                       header = TRUE, sep=",", stringsAsFactors=FALSE)
  ### preprocess dataset
  file_manipulator <- FileManipulator$new()
  data_prepare     <- DataPrepare$new()
  dictionary       <- file_manipulator$loadOrderedDictionary()
  dataset          <- data_prepare$convertAttributeTypes(dataset, dictionary)
  expert           <- Expert$new()
  task             <- list(algorithm = "")
  dataset          <- expert$choosePreprocessing(dataset = dataset, task = task)
  ### call all baseline methods
  file_name <- paste("../thesis_experiments/baseline_methods/models/tpe", substr(files_list[[i]],1,nchar(files_list[[i]])-4),sep="/")
  dir.create(file_name, showWarnings = FALSE)
  file_name <- paste(file_name, "model",sep="/")
  dir.create(file_name, showWarnings = FALSE)
  file_name <- paste(file_name, "model_files",sep="/")
  dir.create(file_name, showWarnings = FALSE)
  ### train knn
  k <- knn_tpe[i]
  opt_parameters <- expand.grid( k = c(k) )
  model      <- caret::train(Class ~ ., data = dataset,
                             method = "knn",
                             tuneGrid = opt_parameters,
                             trControl=trainControl(method="none"))

  model_path = paste(file_name, "tpe_knn.RData", sep ="/")
  save(model, file = model_path)
  ### train nnet
  size   <- nnet_tpe[i,1]
  decay  <- nnet_tpe[i,2]
  opt_parameters <- expand.grid( size = c(size),
                                decay = c(decay))
  model          <- caret::train(Class ~ ., data = dataset,
                             method = "nnet",
                             tuneGrid = opt_parameters,
                             MaxNWts= 100000000,
                             trControl=trainControl(method="none"))
  
  model_path = paste(file_name, "tpe_nnet.RData", sep ="/")
  save(model, file = model_path)
  
  ### train rpart
  cp             <- tree_tpe[i]
  opt_parameters <- expand.grid( cp = c(cp))
  model          <- caret::train(Class ~ ., data = dataset,
                             method = "rpart",
                             tuneGrid = opt_parameters,
                             trControl=trainControl(method="none"))
  
  model_path = paste(file_name, "tpe_tree.RData", sep ="/")
  save(model, file = model_path)
  
  ### train ensemble
  source("../thesis_experiments/baseline_methods/source_code/tpe_ensemble/train.R")
  train(dataset = dataset, file_name = files_list[[i]])
}


