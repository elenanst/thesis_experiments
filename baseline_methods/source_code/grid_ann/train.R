# A function for training an ANN model using grid-search. 

train = function(dataset, file_name) {
  library(caret)
  ## define grid
  fitControl <- trainControl(## 10-fold CV
    method = "repeatedcv",
    number = 10,
    repeats = 1)
  opt_parameters <- expand.grid( size = seq(1,4) ,
                                decay = seq(0,1,0.01))
  
  ## create folds for testing
  folds          <- createFolds(dataset$Class, k = 10, list = TRUE, returnTrain = FALSE)
  split_up       <- lapply(folds, function(ind, dat) dat[ind,], dat = dataset)
  train_split_up <- lapply(folds, function(ind, dat) dat[-ind,], dat = dataset)
  
  accuracies <- list()
  for(iter in 1:length(split_up)) {
    train_dataset       <- train_split_up[[iter]]
    test_dataset        <- split_up[[iter]]
    test_class          <- test_dataset$Class
    test_dataset$Class  <- NULL
    model               <- caret::train(Class ~ ., data = train_dataset,
                                         method = "nnet",
                                         tuneGrid = opt_parameters,
                                         trControl=fitControl)
    
    predictions        <- predict(model, test_dataset)
    cm                 <- confusionMatrix(predictions, test_class)
    accuracies[[iter]] <- cm$overal['Accuracy']
  }
  cross_val_accuracy <- mean(accuracies)
  ## save model
  file_name <- paste("../thesis_experiments/baseline_methods/models/grid",
                     substr(file_name,1, nchar(file_name)-4), sep="/")
  dir.create(file_name, showWarnings = FALSE)
  model_path = paste(file_name, "grid_nnet.RData", sep ="/")
  model$experiment_accuracies <- accuracies
  save(model, file = model_path)
}

