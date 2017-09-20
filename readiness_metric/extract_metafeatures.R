# This script generates training and testing metafeatures.

##  --- set up --- 
rm(list=ls())
setwd("automl")
source("build_script.R")

##  --- generate testing metafeatures --- 
repo       <-"../thesis_experiments/data/datasets/testing"
files_list <- list.files(path = repo,  pattern="*.csv", recursive = TRUE)
for(i in seq(1, length(files_list))) {
  dataset_path <- files_list[[i]]
  dataset      <- read.csv(paste(repo, dataset_path, sep = "/"),
                      header = TRUE, sep=",", stringsAsFactors=FALSE)
  dataset_name <- substr(dataset_path,1,nchar(dataset_path)-4)
  variables    <- names(dataset[sapply(dataset,class) == "character"])
  dataset[, (names(dataset) %in% variables)] <- lapply(as.data.frame(dataset[, (names(dataset) %in% variables)]),
                                                       as.factor)
  file_manipulator <- FileManipulator$new()
  data_prepare     <- DataPrepare$new()
  dictionary       <- file_manipulator$loadOrderedDictionary()
  dataset          <- data_prepare$convertAttributeTypes(dataset, dictionary)
  expert           <- Expert$new()
  dataset          <- expert$choosePreprocessing(dataset=dataset, task = list(inf_action=NA, inf_replace= NA,
                                                                     unknown_action=NA, unknown_replace = NA
                                                                     ,normalize = NA, compress = NA))
  mf1_extractor    <- new('mf1Extractor')
  mf2_extractor    <- new('mf2Extractor', mf1_extractor_ = mf1_extractor)
  meta2features_me <- mf2_extractor$get2MetaFeatures(dataset = dataset)
  if(i == 1 ) {
    total_metafeatures <- meta2features_me
  } else {
    total_metafeatures <- rbind(total_metafeatures,meta2features_me)
  } 
}
rownames(total_metafeatures) <- files_list
write.csv(total_metafeatures, "../thesis_experiments/readiness_metric/testing_metafeatures.csv")

## generate training metafeatures
repo       <-"../thesis_experiments/data/datasets/training"
files_list <- list.files(path = repo,  pattern="*.csv", recursive = TRUE)
for(i in seq(1, length(files_list))) {
  dataset_path <- files_list[[i]]
  dataset      <- read.csv(paste(repo, dataset_path, sep = "/"),
                           header = TRUE, sep=",", stringsAsFactors=FALSE)
  dataset_name <- substr(dataset_path,1,nchar(dataset_path)-4)
  variables    <- names(dataset[sapply(dataset,class) == "character"])
  dataset[, (names(dataset) %in% variables)] <- lapply(as.data.frame(dataset[, (names(dataset) %in% variables)]),
                                                       as.factor)
  file_manipulator <- FileManipulator$new()
  data_prepare     <- DataPrepare$new()
  dictionary       <- file_manipulator$loadOrderedDictionary()
  dataset          <- data_prepare$convertAttributeTypes(dataset, dictionary)
  expert           <- Expert$new()
  mf1_extractor    <- new('mf1Extractor')
  mf2_extractor    <- new('mf2Extractor', mf1_extractor_ = mf1_extractor)
  meta2features_me <- mf2_extractor$get2MetaFeatures(dataset = dataset)
  if(i == 1 ) {
    total_metafeatures <- meta2features_me
  } else {
    total_metafeatures <- rbind(total_metafeatures,meta2features_me)
  } 
}
rownames(total_metafeatures) <- files_list
write.csv(total_metafeatures, "../thesis_experiments/readiness_metric/training_metafeatures.csv")
