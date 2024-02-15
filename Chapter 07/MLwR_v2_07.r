require(neuralnet)
require(kernlab)
require(foreach)
require(doParallel)

##### Chapter 7: Neural Networks and Support Vector Machines -------------------

##### Part 1: Neural Networks -------------------
## Example: Modeling the Strength of Concrete  ----

## Step 2: Exploring and preparing the data ----
# read in data and examine structure
concrete <- read.csv("concrete.csv")
str(concrete)

# custom normalization function
normalize <- function(x) { 
  return((x - min(x)) / (max(x) - min(x)))
}

# apply normalization to entire data frame
concrete_norm <- as.data.frame(lapply(concrete, normalize))

# confirm that the range is now between zero and one
summary(concrete_norm$strength)

# compared to the original minimum and maximum
summary(concrete$strength)

# create training and test data
concrete_train <- concrete_norm[1:773, ]
concrete_test <- concrete_norm[774:1030, ]

## Step 3: Training a model on the data ----
# train the neuralnet model
library(neuralnet)

# Register parallel back-end
registerDoParallel(cores = (detectCores() - 1))

# simple ANN with only a single hidden neuron
set.seed(12345) # to guarantee repeatable results

# Start measuring time for model training
start_time_nn_1 <- Sys.time()

concrete_model <- neuralnet(formula = strength ~ cement + slag +
                              ash + water + superplastic + 
                              coarseagg + fineagg + age,
                              data = concrete_train)

# End measuring time for model training
end_time_nn_1 <- Sys.time()
time_nn_1 <- end_time_nn_1 - start_time_nn_1
print(paste("Time taken for neuralnet 1 hidden neuron:", time_nn_1))

# visualize the network topology
plot(concrete_model)

## Step 4: Evaluating model performance ----

# Start measuring time for computing results
start_time_nn_1_results <- Sys.time()

# obtain model results
model_results <- compute(concrete_model, concrete_test[1:8])

# End measuring time for results
end_time_nn_1_results <- Sys.time()
time_nn_1_results <- end_time_nn_1_results - start_time_nn_1_results
print(paste("Time taken for neuralnet results 1 hidden neuron:", time_nn_1_results))

# obtain predicted strength values
predicted_strength <- model_results$net.result
# examine the correlation between predicted and actual values
cor(predicted_strength, concrete_test$strength)

## Step 5: Improving model performance ----
# a more complex neural network topology with 5 hidden neurons
set.seed(12345) # to guarantee repeatable results

# Start measuring time for model training
start_time_nn_5 <- Sys.time()

concrete_model2 <- neuralnet(strength ~ cement + slag +
                               ash + water + superplastic + 
                               coarseagg + fineagg + age,
                               data = concrete_train, hidden = 5)

# End measuring time for model training
end_time_nn_5 <- Sys.time()
time_nn_5 <- end_time_nn_5 - start_time_nn_5
print(paste("Time taken for neuralnet 5 hidden neurons:", time_nn_5))

# plot the network
plot(concrete_model2)

# Start measuring time for neural net results
start_time_nn_5_results <- Sys.time()

# evaluate the results as we did before
model_results2 <- compute(concrete_model2, concrete_test[1:8])

# End measuring time for results
end_time_nn_5_results <- Sys.time()
time_nn_5_results <- end_time_nn_5_results - start_time_nn_5_results
print(paste("Time taken for neuralnet results 5 hidden neurons:", time_nn_5_results))

predicted_strength2 <- model_results2$net.result
cor(predicted_strength2, concrete_test$strength)

# Stop parallel backend
stopImplicitCluster()



##### Part 2: Support Vector Machines -------------------
## Example: Optical Character Recognition ----

## Step 2: Exploring and preparing the data ----
# read in data and examine structure
letters <- read.csv("letterdata.csv", stringsAsFactors = TRUE)

str(letters)

# divide into training and test data
letters_train <- letters[1:16000, ]
letters_test  <- letters[16001:20000, ]

# Scale the training data
scaled_letters_train <- scale(letters_train[which(names(letters_train) != "letter")])

# Scale the test data using the scaling parameters from the training data
mean_train <- attr(scaled_letters_train, "scaled:center")
std_train <- attr(scaled_letters_train, "scaled:scale")
scaled_letters_test <- scale(letters_test[which(names(letters_test) != "letter")], center = mean_train, scale = std_train)

# Combine the scaled data with the 'letter' variable
letters_train <- data.frame(letter = letters_train$letter, scaled_letters_train)
letters_test <- data.frame(letter = letters_test$letter, scaled_letters_test)

# Check for NAs in the scaled data
if (any(is.na(scaled_letters_train))) {
  print("NA values found in scaled training data")
}

if (any(is.na(scaled_letters_test))) {
  print("NA values found in scaled test data")
}

library(skimr)
skim(letters)
skim(letters_train)
skim(letters_test)

prop.table(table(letters$letter))
prop.table(table(letters_train$letter))
prop.table(table(letters_test$letter))

## Step 3: Training a model on the data ----
# begin by training a simple linear SVM
library(kernlab)

# Register parallel back-end

registerDoParallel(cores = (detectCores() - 1))

# Start measuring time for model training
start_time_svm_1 <- Sys.time()

letter_classifier <- ksvm(letter ~ ., data = letters_train, kernel = "vanilladot")

# End measuring time for model training
end_time_svm_1 <- Sys.time()
time_svm_1 <- end_time_svm_1 - start_time_svm_1
print(paste("Time taken for linear svm classifier:", time_svm_1))

# look at basic information about the model
letter_classifier

## Step 4: Evaluating model performance ----

# Start measuring time for predictions
start_time_svm_1_predict <- Sys.time()

# predictions on testing dataset
letter_predictions <- predict(letter_classifier, letters_test)

# End measuring time for model predictions
end_time_svm_1_predict <- Sys.time()
time_svm_1_predict <- end_time_svm_1_predict - start_time_svm_1_predict
print(paste("Time taken for linear svm classifier predictions:", time_svm_1_predict))

head(letter_predictions)

table(letter_predictions, letters_test$letter)

# look only at agreement vs. non-agreement
# construct a vector of TRUE/FALSE indicating correct/incorrect predictions
agreement <- letter_predictions == letters_test$letter
table(agreement)
prop.table(table(agreement))

# Stop parallel back-end
stopImplicitCluster()

## Step 5: Improving model performance ----

# Register parallel back-end
registerDoParallel(cores = (detectCores() - 1))

set.seed(12345)

# Start measuring time for model training
start_time_svm_2 <- Sys.time()

letter_classifier_rbf <- ksvm(letter ~ ., data = letters_train, kernel = "rbfdot")

# End measuring time for model training
end_time_svm_2 <- Sys.time()
time_svm_2 <- end_time_svm_2 - start_time_svm_2
print(paste("Time taken for Guassian RBF kernel svm classifier:", time_svm_2))

# Start measuring time for predictions
start_time_svm_2_predict <- Sys.time()

letter_predictions_rbf <- predict(letter_classifier_rbf, letters_test)

# End measuring time for model predictions
end_time_svm_2_predict <- Sys.time()
time_svm_2_predict <- end_time_svm_2_predict - start_time_svm_2_predict
print(paste("Time taken for Guassian RBF kernel svm classifier predictions:", time_svm_2_predict))

head(letter_predictions_rbf)
table(letter_predictions_rbf, letters_test$letter)
agreement_rbf <- letter_predictions_rbf == letters_test$letter
table(agreement_rbf)
prop.table(table(agreement_rbf))

# Stop parallel back-end
stopImplicitCluster()

