#
#
# PPDA Project Assignment
# Version 1.2
# 25.11.2019
#
#


# NOTE: The sections of the code correspond
#       to the sections of the report.


### Data cleaning ####
if (FALSE)
{
  rm(list = ls())
  
  #  load the given datasets and combine them (without response)
  data_train <- read.csv(file = "TrainingData.csv")
  data_test <- read.csv(file = "testX.csv")
  data_temp <- rbind(data_train[1:49], data_test)
  
  # replace the indicators for missing values by "NA"
  for (i in 1:dim(data_temp)[1])
    for (j in 1:dim(data_temp)[2])
    {
      if (data_temp[i, j] == "missing" ||
          data_temp[i, j] == "N/A" || data_temp[i, j] == "na")
      {
        data_temp[i, j] = NA
      }
    }
  data <- data_temp
  #save(data, file = "data_NA.RData")
  
  # reset workspace
  rm(list = ls())
  load(file = "data_NA.RData")
  
  # define correct variable class for mvar48
  data[, "mvar48"] <-
    factor(data[, "mvar48"], levels = unique(data[, "mvar48"]))
  
  # derive the number of NAs per variable
  no.NA <-
    apply(
      X = data,
      MARGIN = 2,
      FUN = function(x) {
        sum(is.na(x))
      }
    )
  
  # define a flag mask for selecting all numeric variables
  num_mask <- apply(data, 2,
                    function(x)
                    {
                      if (length(unique(x)) > 50)
                      {
                        return(TRUE)
                      }
                      
                      if (length(unique(x)) <= 50)
                      {
                        return(FALSE)
                      }
                    })
  
  # even though mvar 48 has more than 50 levels it is a categorical variable
  num_mask["mvar48"] <- FALSE
  
  # apply the numeric flag mask on the dataset
  data[, num_mask] <-
    apply(data[, num_mask], 2, function(x) {
      as.numeric(as.character(x))
    })
  
  # eliminate the application key
  features <- data[, -c(1)]
  
  # impute the NAs using missRanger
  library(missRanger)
  features_imp <-
    missRanger(
      data = features,
      seed = 42,
      pmm.k = 3,
      verbose = 2,
      num.trees = 100,
      maxiter = 10
    )
  #save(features_imp, file = "features_imp.RData")
  
  # delete mvar 48 and code mvar 47 correctly
  load("features_imp.RData")
  features_imp <-
    features_imp[, (colnames(features_imp) != "mvar48")]
  features_imp[, "mvar47"] <-
    (as.numeric(features_imp[, "mvar47"]) - 1)
  
  # define all remaining features as numeric
  features_imp <-
    apply(features_imp, 2, function(x) {
      as.numeric(as.character(x))
    })
  
  # re-arange training and test datasets
  train_full <- cbind(data[, 1], features_imp, data[, 50])
  data_train <- read.csv(file = "TrainingData.csv")
  data_train_full <-
    cbind(data_train[, 1], features_imp[1:83000, ], data_train[, "default_ind"])
  colnames(data_train_full)[1] = names(data_train)[1]
  colnames(data_train_full)[49] = names(data_train)[50]
  #save(data_train_full, file = "data_train_full.RData")
  data_test <- read.csv(file = "testX.csv")
  data_test_full <-
    cbind(data_test[, 1], features_imp[((dim(data_train)[1] + 1):dim(features_imp)[1]), ])
  colnames(data_test_full)[1] = names(data_test)[1]
  #save(data_test_full, file = "data_test_full.RData")
  
}# END IF FALSE




#### Model selection ####
if (FALSE)
{
  rm(list = ls())
  
  library(caret)
  library(doParallel)
  
  load("data_train_full.RData")
  
  # draw first step sample of size 10,000 observations
  set.seed(42)
  step1_sample <-
    sample(x = c(1:dim(data_train_full)[1]),
           size = 10000,
           replace = F)
  
  # define response and features
  y <- as.factor(data_train_full[step1_sample, "default_ind"])
  x <- data_train_full[step1_sample, -c(1, dim(data_train_full)[2])]
  
  #summary(x)
  #summary(y)
  
  # configure train control for caret's train function
  # we use 10-fold cv, and search the tuning parameters on a
  # predefined grid, to enhance computation time we allow
  # for parallel computation, before training we use
  # simple up-sampling to cope with class imbalancing.
  tc <-
    trainControl(
      method = "cv",
      number = 10,
      search = "grid",
      verboseIter = T,
      allowParallel = T,
      sampling = "up"
    )
  
  
  result <- NULL
  
  # randomForest model training
  set.seed(42)
  #ranger <- train(x = droplevels(as.data.frame(x)), y =  y, method = "ranger", metric = "Kappa", maximize = T, trControl = tc, preProcess = c("center","scale"), tuneGrid = expand.grid(mtry = seq(1,(dim(x)[2]/3),5), splitrule = c("extratrees","gini"), min.node.size = c(10)))
  #(ranger_kappa <- max(ranger$results[,"Kappa"]))
  #result <- cbind(result, ranger_kappa)
  #save(ranger, file = "ranger_1.RData")
  load("ranger_1.RData")
  
  # initialize parallelization for NN
  try(threats <- floor(detectCores()), silent = T)
  try(cl1 <- makeCluster(threats), silent = T)
  try(registerDoParallel(cl1), silent = T)
  
  # model averaged neural network training
  set.seed(42)
  #avnn <- train(x = x, y =  y, method = "avNNet", metric = "Kappa", maximize = T, trControl = tc, preProcess = c("center","scale"), tuneGrid = expand.grid(size = c(1,3,5), decay = seq(0,0.1,0.02), bag = T), repeats = 10, MaxNWts = 100000)
  #(avnn_kappa <- max(avnn$results[,"Kappa"]))
  #result <- cbind(result, avnn_kappa)
  
  try(stopCluster(cl1), silent = T)
  #save(avnn, file = "avnn_1.RData")
  load("avnn_1.RData")
  
  # k-nearest neighbour training
  set.seed(42)
  #knn <- train(x = x, y =  y, method = "knn",metric = "Kappa",maximize = T,trControl = tc, preProcess = c("center","scale"), tuneGrid = expand.grid(k = c(1:20)))
  #(knn_kappa <- max(knn$results[,"Kappa"]))
  #result <- cbind(result, knn_kappa)
  #save(knn, file = "knn_1.RData")
  load("knn_1.RData")
  
  # ... (euqivalently for all the other models)
  set.seed(42)
  #xgblinear <- train(x = x, y =  y, method = "xgbLinear",metric = "Kappa",maximize = T,trControl = tc, preProcess = c("center","scale"), tuneGrid = expand.grid(eta = 0.3, nrounds = c(100,200),lambda = c(0,0.05,0.1),alpha = c(0,0.05,0.1)))
  #(xgblinear_kappa <- max(xgblinear$results[,"Kappa"]))
  #result <- cbind(result, xgblinear_kappa)
  #save(xgblinear, file = "xgblinear_1.RData")
  load("xgblinear_1.RData")
  
  # ...
  set.seed(42)
  #xgbtree <- train(x = x, y =  y, method = "xgbTree",metric = "Kappa",maximize = T,trControl = tc, preProcess = c("center","scale"),tuneGrid = expand.grid(eta = c(0.1),colsample_bytree = c(0.6,0.8), subsample = c(0.75,1),nrounds = c(100,200,300), min_child_weight = 1, max_depth = c(5,10), gamma = 0))
  #(xgbtree_kappa <- max(xgbtree$results[,"Kappa"]))
  #result <- cbind(result, xgbtree_kappa)
  #save(xgbtree, file = "xgbtree_1.RData")
  load("xgbtree_1.RData")
  
  # ...
  set.seed(42)
  #earth <- train(x = x, y =  y, method = "earth",metric = "Kappa",maximize = T,trControl = tc, preProcess = c("center","scale"), tuneGrid = expand.grid(degree = c(1), nprune = seq(5,floor(dim(x)[2]/2),5)))
  #(earth_kappa <- max(earth$results[,"Kappa"]))
  #result <- cbind(result, earth_kappa)
  #save(earth, file = "earth_1.RData")
  load("earth_1.RData")
  
  # ...
  set.seed(42)
  #rocc <- train(x = x, y =  y, method = "rocc",metric = "Kappa",maximize = T,trControl = tc, preProcess = c("center","scale"), tuneGrid = expand.grid(xgenes = seq(1,floor(dim(x)[2]),5)))
  #(rocc_kappa <- max(rocc$results[,"Kappa"]))
  #result <- cbind(result, rocc_kappa)
  #save(rocc, file = "rocc_1.RData")
  load("rocc_1.RData")
  
  # ...
  set.seed(42)
  #lda <- train(x = x, y =  y, method = "lda2",metric = "Kappa",maximize = T,trControl = tc, preProcess = c("center","scale"), tuneGrid = expand.grid(dimen = c(1:5)))
  #(lda_kappa <- max(lda$results[,"Kappa"], na.rm = T))
  #result <- cbind(result, lda_kappa)
  #save(lda, file = "lda_1.RData")
  load("lda_1.RData")
  
  # ...
  set.seed(42)
  #logitboost <- train(x = x, y =  y, method = "LogitBoost",metric = "Kappa",maximize = T,trControl = tc, preProcess = c("center","scale"), tuneGrid = expand.grid(nIter = seq(11,101,10)))
  #(logitboost_kappa <- max(logitboost$results[,"Kappa"],na.rm = T))
  #result <- cbind(result, logitboost_kappa)
  #save(logitboost, file = "logitboost_1.RData")
  load("logitboost_1.RData")
  
  # ...
  set.seed(42)
  #plr <- train(x = x, y =  y, method = "plr",metric = "Kappa",maximize = T,trControl = tc, preProcess = c("center","scale"),tuneGrid = expand.grid(cp = c("bic","aic"), lambda = seq(0,1e-2,1e-3)))
  #(plr_kappa <- max(plr$results[,"Kappa"],na.rm = T))
  #result <- cbind(result, plr_kappa)
  #save(plr, file = "plr_1.RData")
  load("plr_1.RData")
  
  
  #save(result, file = "result_1.RData")
  load("result_1.RData")
  
  # let's have a look on the results
  result
  
}# END IF FALSE




#### Model fine tuning ####
if (FALSE)
{
  rm(list = ls())
  
  library(caret)
  library(doParallel)
  
  load("data_train_full.RData")
  
  # draw larger sample of 40,000 observations for fine tuning
  set.seed(42)
  step2_sample <-
    sample(x = c(1:dim(data_train_full)[1]),
           size = 40000,
           replace = F)
  
  # define response and features
  y <- as.factor(data_train_full[step2_sample, "default_ind"])
  x <- data_train_full[step2_sample, -c(1, dim(data_train_full)[2])]
  
  #summary(x)
  #summary(y)
  
  # tc initialization as known
  tc <-
    trainControl(
      method = "cv",
      number = 10,
      search = "grid",
      verboseIter = T,
      allowParallel = T,
      sampling = "up"
    )
  
  result <- NULL
  
  # randomForest training on extended tuning grid
  set.seed(42)
  #ranger <- train(x = droplevels(as.data.frame(x)), y = y, method = "ranger", metric = "Kappa", maximize = T, trControl = tc, preProcess = c("center","scale"), tuneGrid = expand.grid(mtry = seq(1,(dim(x)[2]/3),2), splitrule = c("extratrees","gini"), min.node.size = c(10)))
  #(ranger_kappa <- max(ranger$results[,"Kappa"]))
  #result <- cbind(result, ranger_kappa)
  #save(ranger, file = "ranger_2.RData")
  # took 3.5h
  load("ranger_2.RData")
  
  
  # tree based boosting training on extended tuning grid
  set.seed(42)
  #xgbtree <- train(x = x, y =  y, method = "xgbTree",metric = "Kappa",maximize = T,trControl = tc, preProcess = c("center","scale"),tuneGrid =  expand.grid(eta = c(0.01, 0.05, 0.1),colsample_bytree = c(0.6,0.8), subsample = c(0.75,1),nrounds = c(100,200,300,400,500), min_child_weight = 1, max_depth = c(5,10), gamma = 0))
  #(xgbtree_kappa <- max(xgbtree$results[,"Kappa"]))
  #result <- cbind(result, xgbtree_kappa)
  #save(xgbtree, file = "xgbtree_2.RData")
  # took 3.5h
  load("xgbtree_2.RData")
  
  try(threats <- floor(detectCores()), silent = T)
  try(cl1 <- makeCluster(threats), silent = T)
  try(registerDoParallel(cl1), silent = T)
  
  # model averaged neural network training on extended tuning grid
  set.seed(42)
  #avnn <- train(x = x, y =  y, method = "avNNet", metric = "Kappa", maximize = T, trControl = tc, preProcess = c("center","scale"), tuneGrid = expand.grid(size = c(1,3,5,7,9), decay = seq(0,0.1,0.01), bag = T), repeats = 10, MaxNWts = 100000)
  #(avnn_kappa <- max(avnn$results[,"Kappa"]))
  #result <- cbind(result, avnn_kappa)
  # took 14h
  
  try(stopCluster(cl1), silent = T)
  #save(avnn, file = "avnn_2.RData")
  load("avnn_2.RData")
  
  #save(result, file = "result_2.RData")
  
  load("result_2.RData")
  
  # again, the results of the step
  result
  
}# END IF FALSE



#### Model aggregation evaluation ####
if (FALSE)
{
  rm(list = ls())
  
  library(caret)
  
  load("data_train_full.RData")
  
  # draw exactly the same sample as in the model fine tuning phase
  set.seed(42)
  step2_sample <-
    sample(x = c(1:dim(data_train_full)[1]),
           size = 40000,
           replace = F)
  
  # define sample3 as all observations but sample2.
  step3_sample <- c(1:dim(data_train_full)[1])[-step2_sample]
  
  # checksum should be 83,000
  (length(step2_sample) + length(step3_sample))
  
  # again, response and features, now with explicit naming for response
  y <-
    factor(
      data_train_full[step2_sample, "default_ind"],
      levels = c(0, 1),
      labels = c("no_default", "default")
    )
  x <- data_train_full[step2_sample, -c(1, dim(data_train_full)[2])]
  
  # same procedure as every step ..
  tc <-
    trainControl(
      method = "none",
      verboseIter = T,
      allowParallel = T,
      sampling = "up",
      classProbs = T
    )
  
  # retrain the three selected models on optimal hyperparameters ..
  set.seed(42)
  #ranger_3 <- train(x = droplevels(as.data.frame(x)), y = y, method = "ranger", trControl = tc, preProcess = c("center","scale"), tuneGrid = expand.grid(mtry = 5, splitrule = c("extratrees"), min.node.size = c(10)), num.trees = 1000)
  #save(ranger_3, file = "ranger_3.RData")
  load("ranger_3.RData")
  
  set.seed(42)
  #avnn_3 <- train(x = x, y =  y, method = "avNNet", trControl = tc, preProcess = c("center","scale"), tuneGrid = expand.grid(size = c(7), decay = seq(0.02), bag = T), repeats = 50, MaxNWts = 100000)
  #save(avnn_3, file = "avnn_3.RData")
  load("avnn_3.RData")
  
  set.seed(42)
  #xgbtree_3 <- train(x = x, y =  y, method = "xgbTree",trControl = tc, preProcess = c("center","scale"),tuneGrid =  expand.grid(eta = c(0.01),colsample_bytree = c(0.8), subsample = c(1),nrounds = c(500), min_child_weight = 1, max_depth = c(10), gamma = 0))
  #save(xgbtree_3, file = "xgbtree_3.RData")
  load("xgbtree_3.RData")
  
  # redefine features and response for sample3
  y <-
    factor(
      data_train_full[step3_sample, "default_ind"],
      levels = c(0, 1),
      labels = c("no_default", "default")
    )
  x <- data_train_full[step3_sample, -c(1, dim(data_train_full)[2])]
  
  # generate OOS predictions for sample3
  ranger_predict <-
    predict(object = ranger_3,
            newdata = x,
            type = "prob")
  avnn_predict <-
    predict(object = avnn_3,
            newdata = x,
            type = "prob")
  xgbtree_predict <-
    predict(object = xgbtree_3,
            newdata = x,
            type = "prob")
  
  # aggregate the predictions in one matrix
  predict_ <-
    cbind(ranger_predict["default"], avnn_predict["default"], xgbtree_predict["default"])
  
  # define the way of model aggregation,
  # after some experimentation a slight bias towards
  # predicting "default" seems to be reasonable
  agg_fun <- function(x)
  {
    if (mean(x) > 0.45)
    {
      return(1)
    }
    return(0)
  }
  
  # derive model aggregated prediction
  # (generated by an ensemble of 1000 CARTs, 50 neural networks
  #  and 500 boosting base learners.. should be quite robust)
  aggregated_predict <-
    factor(
      apply(predict_, 1, agg_fun),
      levels = c(0, 1),
      labels = c("no_default", "default")
    )
  
  # view OOS results
  confusionMatrix(data = aggregated_predict,
                  reference = y,
                  positive = "default")
  
  
}# END IF FALSE



#### Prediction on test data ####
# this section is structural very equal to the previous section
# we produce OOS predictions for the test features
# after training of the models on all avaliable training observations.
if (FALSE)
{
  rm(list = ls())
  
  library(caret)
  
  load("data_train_full.RData")
  load("data_test_full.RData")
  
  y <-
    factor(
      data_train_full[, "default_ind"],
      levels = c(0, 1),
      labels = c("no_default", "default")
    )
  x <- data_train_full[, -c(1, dim(data_train_full)[2])]
  
  
  tc <-
    trainControl(
      method = "none",
      verboseIter = T,
      allowParallel = T,
      sampling = "up",
      classProbs = T
    )
  
  #ranger
  set.seed(42)
  #ranger_final <- train(x = droplevels(as.data.frame(x)), y = y, method = "ranger", trControl = tc, preProcess = c("center","scale"), tuneGrid = expand.grid(mtry = 5, splitrule = c("extratrees"), min.node.size = c(10)), num.trees = 500) # num trees reduced due to RAM restrictions
  #save(ranger_final, file = "ranger_final.RData")
  load("ranger_final.RData")
  
  #avnn
  set.seed(42)
  #avnn_final <- train(x = x, y =  y, method = "avNNet", trControl = tc, preProcess = c("center","scale"), tuneGrid = expand.grid(size = c(7), decay = seq(0.02), bag = T), repeats = 50, MaxNWts = 100000)
  #save(avnn_final, file = "avnn_final.RData")
  load("avnn_final.RData")
  
  #xgbtree
  set.seed(42)
  #xgbtree_final <- train(x = x, y =  y, method = "xgbTree",trControl = tc, preProcess = c("center","scale"),tuneGrid =  expand.grid(eta = c(0.01),colsample_bytree = c(0.8), subsample = c(1),nrounds = c(500), min_child_weight = 1, max_depth = c(10), gamma = 0))
  #save(xgbtree_final, file = "xgbtree_final.RData")
  load("xgbtree_final.RData")
  
  x <- data_test_full[, -c(1)]
  
  ranger_predict <-
    predict(object = ranger_final,
            newdata = x,
            type = "prob")
  avnn_predict <-
    predict(object = avnn_final,
            newdata = x,
            type = "prob")
  xgbtree_predict <-
    predict(object = xgbtree_final,
            newdata = x,
            type = "prob")
  
  predict_ <-
    cbind(ranger_predict["default"], avnn_predict["default"], xgbtree_predict["default"])
  
  agg_fun <- function(x)
  {
    if (mean(x) > 0.45)
    {
      return(1)
    }
    return(0)
  }
  
  aggregated_predict <-
    factor(
      apply(predict_, 1, agg_fun),
      levels = c(0, 1),
      labels = c("no_default", "default")
    )
  
  submission <- (as.numeric(aggregated_predict) - 1)
  write.table(
    x = submission,
    file = "submission.csv",
    sep = ",",
    row.names = F,
    col.names = F
  )
  
  
}# END IF FALSE
