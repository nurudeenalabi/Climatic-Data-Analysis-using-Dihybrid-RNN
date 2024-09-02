# Load necessary libraries
library(tidyverse)
library(lubridate)
library(zoo)
library(reticulate)
library(tensorflow)
library(keras)
require(ggplotify)
require(gridExtra)
require(cowplot)

# Set working directory
setwd("C:/Users/alabi/OneDrive/Desktop")

# Load the data
climatedata <- read.csv("climatedata_new.csv", header = TRUE, na.strings = "?")

# Ensure 'city' column is present in the dataset
if(!"city" %in% colnames(climatedata)) {
  stop("Column 'city' is not found in the dataset.")
}

# Configure R to use the r-reticulate conda environment
conda_path <- "C:/Users/alabi/Anaconda3/Scripts/conda.exe"
Sys.setenv(RETICULATE_CONDA = conda_path)
use_condaenv("r-reticulate", required = TRUE)
tf$constant("Hello, TensorFlow!")
tf$compat$v1$ConfigProto(gpu_options = tf$compat$v1$GPUOptions(allow_growth = TRUE))

# Preprocess the data
climatedata <- climatedata %>%
  drop_na() %>%
  mutate(across(c(tmin, tmax, sh, ws, rh, ep, rf, sr), as.numeric)) %>%
  mutate(across(c(tmin, tmax, sh, ws, rh, ep, rf, sr), ~ifelse(. == 0, NA, .))) %>%
  drop_na() %>%
  mutate(across(c(tmin, tmax, sh, ws, rh, ep, rf, sr), log))

# Check for any remaining NA or non-numeric values
summary(climatedata)

set.seed(123)
train_indices <- sample(seq_len(nrow(climatedata)), size = 0.8 * nrow(climatedata))
train_data <- climatedata[train_indices, ]
test_data <- climatedata[-train_indices, ]

# Select the relevant columns for features and targets
x_train <- as.matrix(train_data %>% select(tmin, tmax, sh, ws, rh, ep, rf, sr))
y_train <- as.matrix(train_data %>% select(sr, rf, rh))
x_test <- as.matrix(test_data %>% select(tmin, tmax, sh, ws, rh, ep, rf, sr))
y_test <- as.matrix(test_data %>% select(sr, rf, rh))

# Reshape data for LSTM and GRU models
x_train <- array_reshape(x_train, c(nrow(x_train), ncol(x_train), 1))
x_test <- array_reshape(x_test, c(nrow(x_test), ncol(x_test), 1))

# Build LSTM model
build_lstm_model <- function(units = 50, learning_rate = 0.001, optimizer = 'adam', activation = 'tanh', batch_size = 32, epochs = 50) {
  input_layer <- layer_input(shape = c(ncol(x_train), 1))
  lstm_layer <- layer_lstm(units = units, activation = activation, return_sequences = TRUE)(input_layer)
  lstm_layer <- layer_lstm(units = units, activation = activation)(lstm_layer)
  output_layer <- layer_dense(units = 3, activation = 'linear')(lstm_layer)
  
  model <- keras_model(inputs = input_layer, outputs = output_layer)
  model %>% compile(
    loss = 'mean_squared_error',
    optimizer = optimizer,
    metrics = list('mean_absolute_error')
  )
  return(list(model = model, batch_size = batch_size, epochs = epochs))
}

# Build GRU model
build_gru_model <- function(units = 50, learning_rate = 0.001, optimizer = 'adam', activation = 'tanh', batch_size = 32, epochs = 50) {
  input_layer <- layer_input(shape = c(ncol(x_train), 1))
  gru_layer <- layer_gru(units = units, activation = activation, return_sequences = TRUE)(input_layer)
  gru_layer <- layer_gru(units = units, activation = activation)(gru_layer)
  output_layer <- layer_dense(units = 3, activation = 'linear')(gru_layer)
  
  model <- keras_model(inputs = input_layer, outputs = output_layer)
  model %>% compile(
    loss = 'mean_squared_error',
    optimizer = optimizer,
    metrics = list('mean_absolute_error')
  )
  return(list(model = model, batch_size = batch_size, epochs = epochs))
}

# Define a smaller hyperparameter grid for debugging
hyper_grid <- expand.grid(
  units = c(50),
  learning_rate = c(0.0001,0.001, 0.01, 0.1),
  batch_size = c(32,64),  # Smaller batch size for faster execution
  epochs = c(20)       # Fewer epochs for debugging
)

# Define the function to train and evaluate models with cross-validation
train_and_evaluate_cv <- function(build_model_func, hyper_grid, folds = 5) {
  results <- list()
  for (params in 1:nrow(hyper_grid)) {
    model_params <- as.list(hyper_grid[params, ])
    print(paste("Evaluating parameters:", model_params))  # Debugging print
    cv_results <- replicate(folds, {
      indices <- sample(seq_len(nrow(x_train)), size = floor(nrow(x_train) / folds))
      x_train_cv <- x_train[-indices, , , drop = FALSE]
      y_train_cv <- y_train[-indices, , drop = FALSE]
      x_val_cv <- x_train[indices, , , drop = FALSE]
      y_val_cv <- y_train[indices, , drop = FALSE]
      
      model_info <- do.call(build_model_func, model_params)
      model <- model_info$model
      batch_size <- model_info$batch_size
      epochs <- model_info$epochs
      
      history <- model %>% fit(
        x = x_train_cv, y = y_train_cv,
        epochs = epochs, batch_size = batch_size,
        validation_data = list(x_val_cv, y_val_cv), verbose = 0
      )
      evaluation <- model %>% evaluate(x_val_cv, y_val_cv, verbose = 0)
      c(loss = evaluation[[1]], mae = evaluation[[2]])
    })
    cv_results <- t(cv_results)
    colnames(cv_results) <- c("loss", "mae")
    results[[params]] <- c(model_params, colMeans(cv_results))
  }
  
  # Convert results to DataFrame
  results_df <- do.call(rbind, results)
  results_df <- as.data.frame(results_df)
  
  # Set column names
  colnames(results_df) <- c(names(hyper_grid), "loss", "mae")
  results_df
}

# Train and evaluate LSTM models
lstm_results <- train_and_evaluate_cv(build_lstm_model, hyper_grid)
print(lstm_results)

# Extract the best model parameters
best_lstm_result <- lstm_results[which.min(lstm_results$loss), ]
print(best_lstm_result)

# Extract parameters
units <- best_lstm_result$units
learning_rate <- best_lstm_result$learning_rate
batch_size <- best_lstm_result$batch_size
epochs <- best_lstm_result$epochs

# Print the parameters to verify
print(paste("Units:", units))
print(paste("Learning Rate:", learning_rate))
print(paste("Batch Size:", batch_size))
print(paste("Epochs:", epochs))

# Rebuild the best LSTM model with the correct parameters
best_lstm_model_info <- build_lstm_model(
  units = as.numeric(units),
  learning_rate = as.numeric(learning_rate),
  batch_size = as.integer(batch_size),
  epochs = as.integer(epochs)
)

# Print the model summary to verify
best_lstm_model_info$model %>% summary()

# Train the best LSTM model on the full training data
best_lstm_model_info$model %>% fit(
  x = x_train, y = y_train,
  epochs = epochs, batch_size = batch_size,
  validation_split = 0.2, verbose = 0
)

# Evaluate on test data
lstm_eval <- best_lstm_model_info$model %>% evaluate(x_test, y_test, verbose = 0)
best_lstm_preds <- best_lstm_model_info$model %>% predict(x_test)

# Train and evaluate GRU models
gru_results <- train_and_evaluate_cv(build_gru_model, hyper_grid)
print(gru_results)

# Find the best GRU model
best_gru_result <- gru_results[which.min(gru_results$loss), ]
best_gru_result

# Extract parameters correctly
best_gru_result <- as.data.frame(best_gru_result, stringsAsFactors = FALSE)
units <- as.numeric(best_gru_result$units)
learning_rate <- as.numeric(best_gru_result$learning_rate)
batch_size <- as.numeric(best_gru_result$batch_size)
epochs <- as.numeric(best_gru_result$epochs)

# Rebuild the best GRU model with correct parameters
best_gru_model_info <- build_gru_model(
  units = units,
  learning_rate = learning_rate,
  batch_size = batch_size,
  epochs = epochs
)

# Print the model summary to verify
best_gru_model_info$model %>% summary()

# Train the best GRU model on the full training data
best_gru_model_info$model %>% fit(
  x = x_train, y = y_train,
  epochs = epochs, batch_size = batch_size,
  validation_split = 0.2, verbose = 0
)

# Evaluate on test data
gru_eval <- best_gru_model_info$model %>% evaluate(x_test, y_test, verbose = 0)
best_gru_preds <- best_gru_model_info$model %>% predict(x_test)
lstm_eval <- best_lstm_model_info$model %>% evaluate(x_test, y_test, verbose = 0)
best_lstm_preds <- best_lstm_model_info$model %>% predict(x_test)
# Display the evaluation results
list(
  lstm_eval = lstm_eval,
  gru_eval = gru_eval,
  lstm_predictions = best_lstm_preds,
  gru_predictions = best_gru_preds
)

#ANALYSIS ON SOLAR RADIATION, RAINFALL AND RELATIVE HUMIDITY

# Preprocess the data
climatedata <- climatedata %>%
  drop_na() %>%
  mutate(across(c(tmin, tmax, sh, ws, rh, ep, rf, sr), as.numeric)) %>%
  mutate(across(c(tmin, tmax, sh, ws, rh, ep, rf, sr), ~ifelse(. == 0, NA, .))) %>%
  drop_na() %>%
  mutate(across(c(tmin, tmax, sh, ws, rh, ep, rf, sr), log))

# Ensure 'Year' is not included in the features
climatedata <- climatedata %>% select(-Year)

# One-hot encode the 'city' column
climatedata <- climatedata %>%
  mutate(city = factor(city)) %>%
  bind_cols(model.matrix(~ city - 1, data = climatedata))

# Remove the original 'city' column
climatedata <- climatedata %>%
  select(-city)

# Check for any remaining NA or non-numeric values
summary(climatedata)

# Split the data into training and testing sets
set.seed(123)
train_indices <- sample(seq_len(nrow(climatedata)), size = 0.8 * nrow(climatedata))
train_data <- climatedata[train_indices, ]
test_data <- climatedata[-train_indices, ]

# Select the relevant columns for features and targets
x_train <- as.matrix(train_data %>% select(-sr, -rf, -rh))
y_train <- as.matrix(train_data %>% select(sr, rf, rh))
x_test <- as.matrix(test_data %>% select(-sr, -rf, -rh))
y_test <- as.matrix(test_data %>% select(sr, rf, rh))

# Reshape data for LSTM and GRU models
x_train <- array_reshape(x_train, c(nrow(x_train), ncol(x_train), 1))
x_test <- array_reshape(x_test, c(nrow(x_test), ncol(x_test), 1))

# Build LSTM model
build_lstm_model <- function(units = 50, learning_rate = 0.001, optimizer = 'adam', activation = 'tanh', batch_size = 32, epochs = 50) {
  input_layer <- layer_input(shape = c(ncol(x_train), 1))
  lstm_layer <- layer_lstm(units = units, activation = activation, return_sequences = TRUE)(input_layer)
  lstm_layer <- layer_lstm(units = units, activation = activation)(lstm_layer)
  output_layer <- layer_dense(units = 3, activation = 'linear')(lstm_layer)
  
  model <- keras_model(inputs = input_layer, outputs = output_layer)
  model %>% compile(
    loss = 'mean_squared_error',
    optimizer = optimizer,
    metrics = list('mean_absolute_error')
  )
  return(list(model = model, batch_size = batch_size, epochs = epochs))
}

# Build GRU model
build_gru_model <- function(units = 50, learning_rate = 0.001, optimizer = 'adam', activation = 'tanh', batch_size = 32, epochs = 50) {
  input_layer <- layer_input(shape = c(ncol(x_train), 1))
  gru_layer <- layer_gru(units = units, activation = activation, return_sequences = TRUE)(input_layer)
  gru_layer <- layer_gru(units = units, activation = activation)(gru_layer)
  output_layer <- layer_dense(units = 3, activation = 'linear')(gru_layer)
  
  model <- keras_model(inputs = input_layer, outputs = output_layer)
  model %>% compile(
    loss = 'mean_squared_error',
    optimizer = optimizer,
    metrics = list('mean_absolute_error')
  )
  return(list(model = model, batch_size = batch_size, epochs = epochs))
}

# Define a smaller hyperparameter grid for debugging
hyper_grid <- expand.grid(
  units = c(50),
  learning_rate = c(0.0001, 0.001, 0.01, 0.1),
  batch_size = c(32, 64),  # Smaller batch size for faster execution
  epochs = c(20)           # Fewer epochs for debugging
)

# Define the function to train and evaluate models with cross-validation
train_and_evaluate_cv <- function(build_model_func, hyper_grid, folds = 5) {
  results <- list()
  for (params in 1:nrow(hyper_grid)) {
    model_params <- as.list(hyper_grid[params, ])
    print(paste("Evaluating parameters:", model_params))  # Debugging print
    cv_results <- replicate(folds, {
      indices <- sample(seq_len(nrow(x_train)), size = floor(nrow(x_train) / folds))
      x_train_cv <- x_train[-indices, , , drop = FALSE]
      y_train_cv <- y_train[-indices, , drop = FALSE]
      x_val_cv <- x_train[indices, , , drop = FALSE]
      y_val_cv <- y_train[indices, , drop = FALSE]
      
      model_info <- do.call(build_model_func, model_params)
      model <- model_info$model
      batch_size <- model_info$batch_size
      epochs <- model_info$epochs
      
      history <- model %>% fit(
        x = x_train_cv, y = y_train_cv,
        epochs = epochs, batch_size = batch_size,
        validation_data = list(x_val_cv, y_val_cv), verbose = 0
      )
      evaluation <- model %>% evaluate(x_val_cv, y_val_cv, verbose = 0)
      c(loss = evaluation[[1]], mae = evaluation[[2]])
    })
    cv_results <- t(cv_results)
    colnames(cv_results) <- c("loss", "mae")
    results[[params]] <- c(model_params, colMeans(cv_results))
  }
  
  # Convert results to DataFrame
  results_df <- do.call(rbind, results)
  results_df <- as.data.frame(results_df)
  
  # Set column names
  colnames(results_df) <- c(names(hyper_grid), "loss", "mae")
  results_df
}

# Train and evaluate LSTM models
lstm_results <- train_and_evaluate_cv(build_lstm_model, hyper_grid)
print(lstm_results)
# Display the best LSTM result
best_lstm_result <- lstm_results[which.min(lstm_results$loss), ]
print(best_lstm_result)

best_lstm_model_info <- build_lstm_model(
  units = units,
  learning_rate = learning_rate,
  batch_size = batch_size,
  epochs = epochs
)

best_lstm_model_info$model %>% summary()
best_lstm_preds <- best_lstm_model_info$model %>% predict(x_test)

# Train and evaluate GRU models
gru_results <- train_and_evaluate_cv(build_gru_model, hyper_grid)
print(gru_results)

# Find the best GRU model
best_gru_result <- gru_results[which.min(gru_results$loss), ]
best_gru_result

# Extract parameters correctly
best_gru_result <- as.data.frame(best_gru_result, stringsAsFactors = FALSE)
units <- as.numeric(best_gru_result$units)
learning_rate <- as.numeric(best_gru_result$learning_rate)
batch_size <- as.numeric(best_gru_result$batch_size)
epochs <- as.numeric(best_gru_result$epochs)

# Rebuild the best GRU model with the correct parameters
best_gru_model_info <- build_gru_model(
  units = units,
  learning_rate = learning_rate,
  batch_size = batch_size,
  epochs = epochs
)

# Print the model summary to verify
best_gru_model_info$model %>% summary()

# Train the best GRU model on the full training data
best_gru_model_info$model %>% fit(
  x = x_train, y = y_train,
  epochs = epochs, batch_size = batch_size,
  validation_split = 0.2, verbose = 0
)

# Evaluate on test data
gru_eval <- best_gru_model_info$model %>% evaluate(x_test, y_test, verbose = 0)
best_gru_preds <- best_gru_model_info$model %>% predict(x_test)

# Convert predictions back to original scale
best_lstm_preds_original <- exp(best_lstm_preds)
best_gru_preds_original <- exp(best_gru_preds)
y_test_original <- exp(y_test)

# Create data frames for plotting
preds_df <- data.frame(
  Actual_sr = y_test_original[, 1],
  LSTM_Pred_sr = best_lstm_preds_original[, 1],
  GRU_Pred_sr = best_gru_preds_original[, 1],
  Actual_rf = y_test_original[, 2],
  LSTM_Pred_rf = best_lstm_preds_original[, 2],
  GRU_Pred_rf = best_gru_preds_original[, 2],
  Actual_rh = y_test_original[, 3],
  LSTM_Pred_rh = best_lstm_preds_original[, 3],
  GRU_Pred_rh = best_gru_preds_original[, 3]
)

# Plot predictions vs actuals for each target variable
plot_predictions <- function(actual, lstm_pred, gru_pred, title) {
  ggplot(data = preds_df, aes(x = actual)) +
    geom_point(aes(y = lstm_pred, color = "LSTM Predictions")) +
    geom_point(aes(y = gru_pred, color = "GRU Predictions")) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
    labs(title = title, x = "Actual", y = "Predicted") +
    theme_minimal() +
    scale_color_manual(values = c("LSTM Predictions" = "blue", "GRU Predictions" = "red"))
}

p1 <- plot_predictions(preds_df$Actual_sr, preds_df$LSTM_Pred_sr, preds_df$GRU_Pred_sr, "SR Predictions")
p2 <- plot_predictions(preds_df$Actual_rf, preds_df$LSTM_Pred_rf, preds_df$GRU_Pred_rf, "RF Predictions")
p3 <- plot_predictions(preds_df$Actual_rh, preds_df$LSTM_Pred_rh, preds_df$GRU_Pred_rh, "RH Predictions")

# Arrange the plots in a grid
plot_grid(p1, p2, p3, ncol = 1)

# Calculate residuals
residuals_df <- data.frame(
  Residuals_LSTM_sr = preds_df$Actual_sr - preds_df$LSTM_Pred_sr,
  Residuals_GRU_sr = preds_df$Actual_sr - preds_df$GRU_Pred_sr,
  Residuals_LSTM_rf = preds_df$Actual_rf - preds_df$LSTM_Pred_rf,
  Residuals_GRU_rf = preds_df$Actual_rf - preds_df$GRU_Pred_rf,
  Residuals_LSTM_rh = preds_df$Actual_rh - preds_df$LSTM_Pred_rh,
  Residuals_GRU_rh = preds_df$Actual_rh - preds_df$GRU_Pred_rh
)

# Plot residuals for each target variable
plot_residuals <- function(residuals, title) {
  ggplot(data = residuals_df, aes(x = residuals)) +
    geom_histogram(binwidth = 0.1, fill = "blue", alpha = 0.7) +
    labs(title = title, x = "Residuals", y = "Frequency") +
    theme_minimal()
}

r1 <- plot_residuals(residuals_df$Residuals_LSTM_sr, "LSTM SR Residuals")
r2 <- plot_residuals(residuals_df$Residuals_GRU_sr, "GRU SR Residuals")
r3 <- plot_residuals(residuals_df$Residuals_LSTM_rf, "LSTM RF Residuals")
r4 <- plot_residuals(residuals_df$Residuals_GRU_rf, "GRU RF Residuals")
r5 <- plot_residuals(residuals_df$Residuals_LSTM_rh, "LSTM RH Residuals")
r6 <- plot_residuals(residuals_df$Residuals_GRU_rh, "GRU RH Residuals")

# Arrange the residual plots in a grid
plot_grid(r1, r2, r3, r4, r5, r6, ncol = 2)
# Evaluate on test data
gru_eval <- best_gru_model_info$model %>% evaluate(x_test, y_test, verbose = 0)
best_gru_preds <- best_gru_model_info$model %>% predict(x_test)
lstm_eval <- best_lstm_model_info$model %>% evaluate(x_test, y_test, verbose = 0)
best_lstm_preds <- best_lstm_model_info$model %>% predict(x_test)
# Display the evaluation results
list(
  lstm_eval = lstm_eval,
  gru_eval = gru_eval,
  lstm_predictions_original = best_lstm_preds_original,
  gru_predictions_original = best_gru_preds_original
)

