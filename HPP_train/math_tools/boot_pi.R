# This function computes bootstrap confidence intervals for trained machine learning models.

# --- set up ---
library(foreach)
library(doParallel)

# model    : machine learning model
# pdata    : data for prediction
# n        : bootstrap iterations
# p        : confidence level
# enableLog: apply log-transformation
# cores    : number of cores 
# return   : data.frame of predictions and intervals
boot_pi = function(model, pdata, n, p, enableLog = TRUE, cores = 2) { 
  optParameters <- model$bestTune
  odata         <- model$trainingData
  lp            <- (1 - p) / 2
  up            <- 1 - lp
  set.seed(1)
  seeds <- round(runif(n, 1, 1000), 0)
  registerDoParallel(cores = cores)
  boot_y <- foreach(i = 1:n, .combine = rbind) %dopar% {
    set.seed(seeds[i])
    # choose bootstrap sample
    bdata <- odata[sample(seq(nrow(odata)), size = nrow(odata), replace = TRUE), ]
    # perform bootstrap prediction
    if(enableLog) {
      model_current <- caret::train(log10(.outcome)~ ., data = bdata,
                                    method = model$method,
                                    tuneGrid = optParameters,
                                    trControl=trainControl(method="none"))
      bpred         <- 10^predict(model_current, pdata)
    } else {
      model_current <- caret::train(.outcome~ ., data = bdata,
                                    method = model$method,
                                    tuneGrid = optParameters,
                                    trControl=trainControl(method="none"))
      bpred         <- predict(model_current, newdata = pdata)
    }
  }
  # calculate confidence intervals
  boot_ci   <- t(apply(boot_y, 2, quantile, c(lp, up))) 
  predicted <- predict(model, newdata = pdata)
  str(predicted)
  if(enableLog) predicted <- 10^predicted
  return(data.frame(pred = predicted, lower = boot_ci[, 1] , upper = boot_ci[, 2] ))
}