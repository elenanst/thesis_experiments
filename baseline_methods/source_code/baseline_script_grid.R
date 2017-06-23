# This script is used to create benchmark data for grid-search optimized models. Benchmark data include the performance of knn, nnet, rpart and forward model ensembleon testing datasets.

## set up 
rm(list=ls())
setwd("automl")
source("build_script.R")


## load data 
datasets_path <- "workspace/datasets_repo"
files_list = list.files(path = datasets_path,
                        pattern="*.csv", recursive = TRUE)

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
  source("../thesis_experiments/baseline_methods/source_code/grid_ann/train.R")
  train(dataset = dataset, file_name = files_list[[i]])
  source("../thesis_experiments/baseline_methods/source_code/grid_knn/train.R")
  train(dataset = dataset, file_name = files_list[[i]])
  source("../thesis_experiments/baseline_methods/source_code/grid_tree/train.R")
  train(dataset = dataset, file_name = files_list[[i]])
  source("../thesis_experiments/baseline_methods/source_code/grid_ensemble/train.R")
  train(dataset = dataset, file_name = files_list[[i]])
}


