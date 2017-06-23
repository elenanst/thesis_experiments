# This function computes bootstrap confidence intervals
library(foreach)
library(doParallel)
boot_pi = function(model, pdata, n, p, enableLog = TRUE) { 
  registerDoParallel(cores = 4)
  params_list <- list()
  params <- names(model$bestTune)
  x<- params[1]
  optParameters <- data.frame( param= 2)
  if(length(params)>1) {
    for(i in 2:length(params)) {
      optParameters <- cbind(optParameters, data.frame( temp = model$bestTune[params[i]])) 
    }
  }
  names(optParameters) <- params
  odata <- model$trainingData
  lp <- (1 - p) / 2
  up <- 1 - lp
  set.seed(1)
  seeds <- round(runif(n, 1, 1000), 0)
  boot_y <- foreach(i = 1:n, .combine = rbind) %dopar% {
    set.seed(seeds[i])
    bdata <- odata[sample(seq(nrow(odata)), size = nrow(odata), replace = TRUE), ]
    if(enableLog) {
      model_current <- caret::train(log10(.outcome)~ ., data = bdata,
                                    method = model$method,
                                    tuneGrid = optParameters,
                                    trControl=trainControl(method="none"))
      bpred <- predict(model_current, newdata = pdata)
      bpred <-10^bpred
    } else {
      model_current <- caret::train(.outcome~ ., data = bdata,
                                    method = model$method,
                                    tuneGrid = optParameters,
                                    trControl=trainControl(method="none"))
      bpred <- predict(model_current, newdata = pdata)
      bpred <-bpred
    }
    bpred
  }
  boot_ci <- t(apply(boot_y, 2, quantile, c(lp, up))) 
  if(enableLog) { 
    predicted <- 10^predict(model, newdata = pdata)
  }
  else {
    predicted <- predict(model, newdata = pdata)
  }
  return(data.frame(pred = predicted, lower = boot_ci[, 1] , upper = boot_ci[, 2] ))
}