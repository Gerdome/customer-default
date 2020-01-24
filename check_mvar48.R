#
# This subfile just examines the effect of mvar48.
# Its structure follows directly the structure of the main file.
#


### Data cleaning ####
if (FALSE)
{
  rm(list = ls())
  data_train <- read.csv(file = "TrainingData.csv")
  data_test <- read.csv(file = "testX.csv")
  data_temp <- rbind(data_train[1:49], data_test)
  
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
} #END IF FALSE

if (FALSE)
{
  rm(list = ls())
  load(file = "data_NA.RData")
  data[, "mvar48"] <-
    factor(data[, "mvar48"], levels = unique(data[, "mvar48"]))
  no.NA <-
    apply(
      X = data,
      MARGIN = 2,
      FUN = function(x) {
        sum(is.na(x))
      }
    )
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
  num_mask["mvar48"] <- FALSE
  data[, num_mask] <-
    apply(data[, num_mask], 2, function(x) {
      as.numeric(as.character(x))
    })
  features <- data[, -c(1)]
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
  load("features_imp.RData")
  features_imp <- features_imp[, (colnames(features_imp) != "mvar48")]
  features_imp[, "mvar47"] <- (as.numeric(features_imp[, "mvar47"]) - 1)
  features_imp <-
    apply(features_imp, 2, function(x) {
      as.numeric(as.character(x))
    })
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
  
  set.seed(42)
  step1_sample <-
    sample(x = c(1:dim(data_train_full)[1]),
           size = 10000,
           replace = F)
  
  
  y <- as.factor(data_train_full[step1_sample, "default_ind"])
  x <- data_train_full[step1_sample, -c(1, dim(data_train_full)[2])]
  
  data_train <- read.csv(file = "TrainingData.csv")
  mvar48 <-
    factor(data_train[step1_sample, "mvar48"], levels = unique(data_train[step1_sample, "mvar48"]))
  x <- cbind(x, mvar48)
  
  #summary(x)
  #summary(y)
  
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
  
  set.seed(42)
  xgbtree <-
    train(
      x = x,
      y =  y,
      method = "xgbTree",
      metric = "Kappa",
      maximize = T,
      trControl = tc,
      preProcess = c("center", "scale"),
      tuneGrid = expand.grid(
        eta = c(0.1),
        colsample_bytree = c(0.6, 0.8),
        subsample = c(0.75, 1),
        nrounds = c(100, 200, 300),
        min_child_weight = 1,
        max_depth = c(5, 10),
        gamma = 0
      )
    )
  (xgbtree_kappa <- max(xgbtree$results[, "Kappa"]))
  result <- cbind(result, xgbtree_kappa)
  save(xgbtree, file = "xgbtree_with_mvar48.RData")
  
  #save(result, file = "result_with_mvar48.RData")
  
  load("result_with_mvar48.RData")
  result
  
  load("result_1.RData")
  result
  
  length(unique(read.csv(file = "TrainingData.csv")[, "mvar48"]))
  length(unique(rbind(
    read.csv(file = "TrainingData.csv")[-50],
    read.csv(file = "testX.csv")
  )   [, "mvar48"]))
  
  # -> increase of predictive power is only minor
  # -> effect of mvar48 might already be partly captured in
  #    NA imputation phase
  # -> danger of severe impact on OOS test predictions
  #    due to unknown levels in test features
  # -> no reasonable way of utilizing mvar48 in most non-treebased
  #    algorithms
  #  ------------------
  # --> exclude mvar48
  
}# END IF FALSE
