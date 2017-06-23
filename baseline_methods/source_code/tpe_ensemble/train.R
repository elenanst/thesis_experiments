# A function for training a forward-model selection ensemble. It is assumed that individual models have already been trained.

train = function(dataset, file_name) {
  
  # define directory containing models
  file_name <- substr(file_name,1, nchar(file_name)-4)
  models_dir       <- paste("../thesis_experiments/baseline_methods/models/tpe", file_name,"model/model_files", sep = "/")
  
  # 10-fold cross-validation
  folds <- createFolds(dataset$Class, k = 10, list = TRUE, returnTrain = FALSE)
  split_up <- lapply(folds, function(ind, dat) dat[ind,], dat = dataset)
  train_split_up <- lapply(folds, function(ind, dat) dat[-ind,], dat = dataset)
  accuracies <- list()
  for(iter in 1:length(split_up)) {
    # get a copy of original models for each fold
    str(models_dir)
    current_fold <- paste(models_dir, "folds", sep="_")
    dir.create(current_fold, showWarnings = FALSE)
    dir.create(paste(paste(current_fold, "fold",sep="/"), iter, sep = "_"), showWarnings = FALSE)
    str(paste(paste(current_fold, "fold",sep="/"), iter, sep = "_"))
    models_to_copy_or <- as.list(list.files(path = models_dir, pattern="*.RData"))
    models_to_copy <- lapply(models_to_copy_or, function(x) paste(models_dir, x,sep="/"))
    copy_dir_or <- paste(paste(current_fold, "fold",sep="/"), iter, sep = "_")
    copy_dir <- paste(copy_dir_or, "model",sep="/")
    dir.create(copy_dir,showWarnings = FALSE)
    copy_dir <- paste(copy_dir, "model_files",sep="/")
    dir.create(copy_dir,showWarnings = FALSE)
    str(copy_dir)
    str(models_to_copy_or)
    target_dir <- lapply(models_to_copy_or, function(x) paste(copy_dir,x,sep="/"))
    file.copy(from=unlist(models_to_copy), to= unlist(target_dir),
              copy.mode = TRUE,recursive=FALSE)
    train_dataset <- train_split_up[[iter]]
    test_dataset  <- split_up[[iter]]
    test_class <- test_dataset$Class
    test_dataset$Class <- NULL
    classifier       <- GenericClassifier$new()
    ensembler        <- Ensembler$new(M_=3,perc_initial_=0)
    included_models  <- ensembler$ensemble(classifier = classifier, test_dataset = train_dataset, performance_metric = "Accuracy",
                     project_dir = copy_dir_or)
    # delete all unselected models
    total_models     <- classifier$getModels(project_dir = models_dir)
    unused_models    <- setdiff(total_models, included_models)
    models_to_delete <- as.vector(unused_models)
    unlink(models_to_delete)
    # calculate accuracy of ensemble
    datasets<-list()
    for(d in 1:length(included_models)) 
    {
      datasets[[d]] <- test_dataset
    }
    predictions      <- ensembler$getEnsemblePredictions(datasets =datasets,
                                                      type = "raw", project_dir = copy_dir_or)
    str(predictions)
    str(test_class)
    cat(length(predictions))
    cat(length(test_class))
    cm <- confusionMatrix(predictions, test_class)
    accuracies[[iter]] <- cm$overal['Accuracy']
  }
  cross_val_accuracy <- mean(accuracies)
  # save information
  save(accuracies, file = paste(models_dir, "enseble_accuracies.RData", sep = "/"))
}