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
climatedata <- climatedata %>%
  select(year, city, sr, tmin, tmax, rh, sh, ep, ws, rf) %>%
  mutate(across(c(sr, tmin, tmax, rh, sh, ep, ws, rf), as.numeric)) %>%
  drop_na() %>%
  mutate(across(c(sr, tmin, tmax, rh, sh, ep, ws, rf), ~ifelse(. == 0, NA, .))) %>%
  drop_na() %>%
  mutate(across(c(sr, tmin, tmax, rh, sh, ep, ws, rf), log))
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
# Data preparation
set.seed(123)
train_indices <- sample(seq_len(nrow(climatedata)), size = 0.8 * nrow(climatedata))
train_data <- climatedata[train_indices, ]
test_data <- climatedata[-train_indices, ]

x_train <- as.matrix(train_data %>% select(tmin, tmax, sh, ws, rh, ep, rf))
y_train <- as.matrix(train_data$sr)
x_test <- as.matrix(test_data %>% select(tmin, tmax, sh, ws, rh, ep, rf))
y_test <- as.matrix(test_data$sr)

x_train <- array_reshape(x_train, c(nrow(x_train), ncol(x_train), 1))
x_test <- array_reshape(x_test, c(nrow(x_test), ncol(x_test), 1))
# Define custom metrics
rmse <- function(y_true, y_pred) {
  K <- backend()
  return(K$sqrt(K$mean(K$square(y_pred - y_true), axis = as.integer(-1))))
}
rsquare <- function(y_true, y_pred) {
  K <- backend()
  ss_res <- K$sum(K$square(y_true - y_pred))
  ss_tot <- K$sum(K$square(y_true - K$mean(y_true)))
  return(1 - ss_res / (ss_tot + K$epsilon()))
}
# Build LSTM model function with fixed parameters
build_lstm_model <- function(units = 200, learning_rate = 0.00001, optimizer = 'adam', activation = 'relu', recurrent_activation = 'gelu', batch_size = 128, epochs = 200) {
  input_layer <- layer_input(shape = c(ncol(x_train), 1))
  lstm_layer <- layer_lstm(units = units, activation = activation, return_sequences = TRUE)(input_layer)
  lstm_layer <- layer_lstm(units = units, activation = activation)(lstm_layer)
  output_layer <- layer_dense(units = 1, activation = 'linear')(lstm_layer)
  
  model <- keras_model(inputs = input_layer, outputs = output_layer)
  optimizer <- optimizer_adam(learning_rate = learning_rate)
  model %>% compile(
    loss = 'mean_squared_error',
    optimizer = optimizer,
    metrics = list('mean_absolute_error', custom_metric("rmse", rmse), custom_metric("rsquare", rsquare))
  )
  return(list(model = model, batch_size = batch_size, epochs = epochs))
}
# Build GRU model function with fixed parameters
build_gru_model <- function(units = 200, learning_rate = 0.00001, optimizer = 'adam', activation = 'tanh', recurrent_activation = 'hard_tanh', batch_size = 128, epochs = 200) {
  input_layer <- layer_input(shape = c(ncol(x_train), 1))
  gru_layer <- layer_gru(units = units, activation = activation, return_sequences = TRUE)(input_layer)
  gru_layer <- layer_gru(units = units, activation = activation)(gru_layer)
  output_layer <- layer_dense(units = 1, activation = 'linear')(gru_layer)
  
  model <- keras_model(inputs = input_layer, outputs = output_layer)
  optimizer <- optimizer_adam(learning_rate = learning_rate)
  model %>% compile(
    loss = 'mean_squared_error',
    optimizer = optimizer,
    metrics = list('mean_absolute_error', custom_metric("rmse", rmse), custom_metric("rsquare", rsquare))
  )
  return(list(model = model, batch_size = batch_size, epochs = epochs))
}
# Train and evaluate LSTM model
best_lstm_model_info <- build_lstm_model()
best_lstm_model_info$model %>% fit(
  x = x_train, y = y_train,
  epochs = best_lstm_model_info$epochs, batch_size = best_lstm_model_info$batch_size,
  validation_split = 0.2, verbose = 0
)
print(best_lstm_model_info$model)
lstm_eval <- best_lstm_model_info$model %>% evaluate(x_test, y_test, verbose = 0)
print(lstm_eval)
best_lstm_preds <- best_lstm_model_info$model %>% predict(x_test)
print(best_lstm_preds)
###### Train and evaluate GRU model
best_gru_model_info <- build_gru_model()
best_gru_model_info$model %>% fit(
  x = x_train, y = y_train,
  epochs = best_gru_model_info$epochs, batch_size = best_gru_model_info$batch_size,
  validation_split = 0.2, verbose = 0
)
print(best_gru_model_info$model)
gru_eval <- best_gru_model_info$model %>% evaluate(x_test, y_test, verbose = 0)
print(gru_eval)
best_gru_preds <- best_gru_model_info$model %>% predict(x_test)
print(best_gru_preds)
###### Reverse log transformation
y_test_exp <- exp(y_test)
best_lstm_preds_exp <- exp(best_lstm_preds)
best_gru_preds_exp <- exp(best_gru_preds)
print(best_lstm_preds_exp)
print(best_gru_preds_exp)
###### Plotting
plot_df_lstm <- data.frame(Actual = y_test_exp, Predicted = best_lstm_preds_exp)
lstm_plot <- ggplot(plot_df_lstm, aes(x = seq_along(Actual))) +
  geom_line(aes(y = Actual, color = "Actual")) +
  geom_line(aes(y = Predicted, color = "Predicted")) +
  labs(title = "LSTM: Actual vs Predicted", x = "Index", y = "Values") +
  scale_color_manual(values = c("Actual" = "red", "Predicted" = "blue"))
plot_df_gru <- data.frame(Actual = y_test_exp, Predicted = best_gru_preds_exp)
gru_plot <- ggplot(plot_df_gru, aes(x = seq_along(Actual))) +
  geom_line(aes(y = Actual, color = "Actual")) +
  geom_line(aes(y = Predicted, color = "Predicted")) +
  labs(title = "GRU: Actual vs Predicted", x = "Index", y = "Values") +
  scale_color_manual(values = c("Actual" = "green", "Predicted" = "black"))

plot_grid(lstm_plot, gru_plot, nrow = 1, ncol = 2)

##### Report results for each city
results_by_city <- climatedata %>%
  group_by(city) %>%
  group_modify(~ {
    lstm_mae <- numeric()
    gru_mae <- numeric()
    lstm_rmse <- numeric()
    gru_rmse <- numeric()
    lstm_rsquare <- numeric()
    gru_rsquare <- numeric()
    lstm_preds <- numeric()
    gru_preds <- numeric()
    
    k <- 20  # Number of cross-validation folds
    for (i in 1:k) {
      fold_size <- floor(nrow(.x) / k)
      test_indices <- ((i - 1) * fold_size + 1):(i * fold_size)
      test_fold <- .x[test_indices, ]
      train_fold <- .x[-test_indices, ]
      
      x_train_cv <- as.matrix(train_fold %>% select(tmin, tmax, sh, ws, rh, ep, rf))
      y_train_cv <- as.matrix(train_fold$sr)
      x_test_cv <- as.matrix(test_fold %>% select(tmin, tmax, sh, ws, rh, ep, rf))
      y_test_cv <- as.matrix(test_fold$sr)
      
      x_train_cv <- array_reshape(x_train_cv, c(nrow(x_train_cv), ncol(x_train_cv), 1))
      x_test_cv <- array_reshape(x_test_cv, c(nrow(x_test_cv), ncol(x_test_cv), 1))
      
      lstm_model_cv <- best_lstm_model_info$model
      lstm_model_cv %>% fit(x_train_cv, y_train_cv, epochs = best_lstm_model_info$epochs, batch_size = best_lstm_model_info$batch_size, verbose = 0)
      lstm_eval_cv <- lstm_model_cv %>% evaluate(x_test_cv, y_test_cv, verbose = 0)
      lstm_preds_cv <- lstm_model_cv %>% predict(x_test_cv)
      lstm_preds_exp_cv <- exp(lstm_preds_cv)
      
      gru_model_cv <- best_gru_model_info$model
      gru_model_cv %>% fit(x_train_cv, y_train_cv, epochs = best_gru_model_info$epochs, batch_size = best_gru_model_info$batch_size, verbose = 0)
      gru_eval_cv <- gru_model_cv %>% evaluate(x_test_cv, y_test_cv, verbose = 0)
      gru_preds_cv <- gru_model_cv %>% predict(x_test_cv)
      gru_preds_exp_cv <- exp(gru_preds_cv)
      
      y_test_exp_cv <- exp(y_test_cv)
      
      lstm_mae <- c(lstm_mae, mean(abs(lstm_preds_exp_cv - y_test_exp_cv)))
      gru_mae <- c(gru_mae, mean(abs(gru_preds_exp_cv - y_test_exp_cv)))
      lstm_rmse <- c(lstm_rmse, sqrt(mean((lstm_preds_exp_cv - y_test_exp_cv)^2)))
      gru_rmse <- c(gru_rmse, sqrt(mean((gru_preds_exp_cv - y_test_exp_cv)^2)))
      lstm_rsquare <- c(lstm_rsquare, cor(lstm_preds_exp_cv, y_test_exp_cv)^2)
      gru_rsquare <- c(gru_rsquare, cor(gru_preds_exp_cv, y_test_exp_cv)^2)
      lstm_preds <- c(lstm_preds, lstm_preds_exp_cv)
      gru_preds <- c(gru_preds, gru_preds_exp_cv)
    }
    
    lstm_mae_mean <- mean(lstm_mae)
    gru_mae_mean <- mean(gru_mae)
    lstm_rmse_mean <- mean(lstm_rmse)
    gru_rmse_mean <- mean(gru_rmse)
    lstm_rsquare_mean <- mean(lstm_rsquare)
    gru_rsquare_mean <- mean(gru_rsquare)
    
    return(tibble(lstm_mae = lstm_mae_mean, gru_mae = gru_mae_mean, lstm_rmse = lstm_rmse_mean, gru_rmse = gru_rmse_mean, lstm_rsquare = lstm_rsquare_mean, gru_rsquare = gru_rsquare_mean, lstm_preds = list(lstm_preds), gru_preds = list(gru_preds)))
  })
print(results_by_city)
##### Load necessary libraries
library(dplyr)
library(keras)
library(tensorflow)
# Assume 'climatedata' is your original dataset
# Split the dataset by city
ilorin_data <- climatedata %>% filter(city == "ILORIN")
sokoto_data <- climatedata %>% filter(city == "SOKOTO")
enugu_data <- climatedata %>% filter(city == "ENUGU")
preprocess_data <- function(data) {
  data <- data %>%
    mutate(across(c(sr, tmin, tmax, rh, sh, ep, ws, rf), as.numeric)) %>%
    drop_na() %>%
    mutate(across(c(sr, tmin, tmax, rh, sh, ep, ws, rf), ~ifelse(. == 0, NA, .))) %>%
    drop_na() %>%
    mutate(across(c(sr, tmin, tmax, rh, sh, ep, ws, rf), log))
  
  # Split data into train and test sets
  set.seed(123)
  train_indices <- sample(seq_len(nrow(data)), size = 0.8 * nrow(data))
  train_data <- data[train_indices, ]
  test_data <- data[-train_indices, ]
  
  # Select the relevant columns for features and target
  x_train <- as.matrix(train_data %>% select(tmin, tmax, sh, ws, rh, ep, rf))
  y_train <- as.matrix(train_data$sr)
  x_test <- as.matrix(test_data %>% select(tmin, tmax, sh, ws, rh, ep, rf))
  y_test <- as.matrix(test_data$sr)
  
  # Reshape data for LSTM and GRU models
  x_train <- array_reshape(x_train, c(nrow(x_train), ncol(x_train), 1))
  x_test <- array_reshape(x_test, c(nrow(x_test), ncol(x_test), 1))
  
  list(x_train = x_train, y_train = y_train, x_test = x_test, y_test = y_test)
}
# Preprocess data for each city
ilorin_processed <- preprocess_data(ilorin_data)
sokoto_processed <- preprocess_data(sokoto_data)
enugu_processed <- preprocess_data(enugu_data)
# Define custom metrics
rmse <- function(y_true, y_pred) {
  K <- backend()
  return(K$sqrt(K$mean(K$square(y_pred - y_true), axis = as.integer(-1))))
}
rsquare <- function(y_true, y_pred) {
  K <- backend()
  ss_res <- K$sum(K$square(y_true - y_pred))
  ss_tot <- K$sum(K$square(y_true - K$mean(y_true)))
  return(1 - ss_res / (ss_tot + K$epsilon()))
}

build_lstm_model <- function(units = 50, learning_rate = 0.001, batch_size = 32, epochs = 100, recurrent_activation = 'sigmoid') {
  model <- keras_model_sequential() %>%
    layer_lstm(units, activation = 'relu', recurrent_activation = recurrent_activation, input_shape = c(7, 1), return_sequences = TRUE) %>%
    layer_lstm(units, activation = 'relu', recurrent_activation = recurrent_activation) %>%
    layer_dense(1, activation = 'linear')
  
  model %>% compile(
    loss = 'mean_squared_error',
    optimizer = optimizer_adam(lr = learning_rate),
    metrics = list('mean_absolute_error', custom_metric("rmse", rmse), custom_metric("rsquare", rsquare))
  )
  
  model
}
build_gru_model <- function(units = 50, learning_rate = 0.001, batch_size = 32, epochs = 100, recurrent_activation = 'sigmoid') {
  model <- keras_model_sequential() %>%
    layer_gru(units, activation = 'relu', recurrent_activation = recurrent_activation, input_shape = c(7, 1), return_sequences = TRUE) %>%
    layer_gru(units, activation = 'relu', recurrent_activation = recurrent_activation) %>%
    layer_dense(1, activation = 'linear')
  
  model %>% compile(
    loss = 'mean_squared_error',
    optimizer = optimizer_adam(lr = learning_rate),
    metrics = list('mean_absolute_error', custom_metric("rmse", rmse), custom_metric("rsquare", rsquare))
  )
  
  model
}
# Modify the train_model function to set run_eagerly=True
train_model <- function(model, x_train, y_train, x_test, y_test, epochs = 100, batch_size = 32) {
  model %>% compile(
    loss = 'mean_squared_error',
    optimizer = optimizer_adam(lr = 0.001),
    run_eagerly = TRUE  # Enable eager execution
  )
  
  history <- model %>% fit(
    x_train, y_train,
    epochs = epochs,
    batch_size = batch_size,
    validation_data = list(x_test, y_test),
    verbose = 2
  )
  
  return(history)
}
# Train and evaluate LSTM and GRU models for ILORIN
# ilorin_lstm <- build_lstm_model(recurrent_activation = 'sigmoid')
# Build and train LSTM model with eager execution
ilorin_lstm <- build_lstm_model(recurrent_activation = 'sigmoid')
ilorin_lstm %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam(lr = 0.001),
  metrics = c('mean_squared_error')
)

ilorin_lstm_history <- ilorin_lstm %>% fit(
  x = ilorin_processed$x_train,
  y = ilorin_processed$y_train,
  epochs = 100,
  batch_size = 32,
  validation_data = list(ilorin_processed$x_test, ilorin_processed$y_test),
  verbose = 2,
  run_eagerly = TRUE
)

ilorin_lstm_history <- train_model(ilorin_lstm, ilorin_processed$x_train, ilorin_processed$y_train, ilorin_processed$x_test, ilorin_processed$y_test)

ilorin_gru <- build_gru_model(recurrent_activation = 'sigmoid')
ilorin_gru_history <- train_model(ilorin_gru, ilorin_processed$x_train, ilorin_processed$y_train, ilorin_processed$x_test, ilorin_processed$y_test)

# Train and evaluate LSTM and GRU models for SOKOTO
sokoto_lstm <- build_lstm_model(recurrent_activation = 'sigmoid')
sokoto_lstm_history <- train_model(sokoto_lstm, sokoto_processed$x_train, sokoto_processed$y_train, sokoto_processed$x_test, sokoto_processed$y_test)

sokoto_gru <- build_gru_model(recurrent_activation = 'sigmoid')
sokoto_gru_history <- train_model(sokoto_gru, sokoto_processed$x_train, sokoto_processed$y_train, sokoto_processed$x_test, sokoto_processed$y_test)

# Train and evaluate LSTM and GRU models for ENUGU
enugu_lstm <- build_lstm_model(recurrent_activation = 'sigmoid')
enugu_lstm_history <- train_model(enugu_lstm, enugu_processed$x_train, enugu_processed$y_train, enugu_processed$x_test, enugu_processed$y_test)

enugu_gru <- build_gru_model(recurrent_activation = 'sigmoid')
enugu_gru_history <- train_model(enugu_gru, enugu_processed$x_train, enugu_processed$y_train, enugu_processed$x_test, enugu_processed$y_test)
plot_predictions <- function(model, x_test, y_test, title) {
  predictions <- model %>% predict(x_test)
  
  # Transform predictions and actual values back to original scale
  predictions_original <- exp(predictions)
  y_test_original <- exp(y_test)
  
  plot_data <- data.frame(Index = 1:length(y_test), Actual = y_test_original, Predicted = predictions_original)
  
  ggplot(plot_data, aes(x = Index)) +
    geom_line(aes(y = Actual, color = 'Actual')) +
    geom_line(aes(y = Predicted, color = 'Predicted')) +
    labs(title = title, x = 'Index', y = 'Values') +
    scale_color_manual(values = c('Actual' = 'red', 'Predicted' = 'blue'))
}
plot_residuals <- function(model, x_test, y_test, title) {
  predictions <- model %>% predict(x_test)
  
  # Transform predictions and actual values back to original scale
  predictions_original <- exp(predictions)
  y_test_original <- exp(y_test)
  
  residuals <- y_test_original - predictions_original
  residual_data <- data.frame(Index = 1:length(y_test), Residuals = residuals)
  
  ggplot(residual_data, aes(x = Index, y = Residuals)) +
    geom_point(color = 'blue') +
    labs(title = title, x = 'Index', y = 'Residuals')
}
# Plot predictions vs actual and residuals for ILORIN LSTM
plot_predictions(ilorin_lstm, ilorin_processed$x_test, ilorin_processed$y_test, 'Ilorin LSTM: Actual vs Predicted')
plot_residuals(ilorin_lstm, ilorin_processed$x_test, ilorin_processed$y_test, 'Ilorin LSTM: Residuals')

# Plot predictions vs actual and residuals for ILORIN GRU
plot_predictions(ilorin_gru, ilorin_processed$x_test, ilorin_processed$y_test, 'Ilorin GRU: Actual vs Predicted')
plot_residuals(ilorin_gru, ilorin_processed$x_test, ilorin_processed$y_test, 'Ilorin GRU: Residuals')

# Plot predictions vs actual and residuals for SOKOTO LSTM
plot_predictions(sokoto_lstm, sokoto_processed$x_test, sokoto_processed$y_test, 'Sokoto LSTM: Actual vs Predicted')
plot_residuals(sokoto_lstm, sokoto_processed$x_test, sokoto_processed$y_test, 'Sokoto LSTM: Residuals')

# Plot predictions vs actual and residuals for SOKOTO GRU
plot_predictions(sokoto_gru, sokoto_processed$x_test, sokoto_processed$y_test, 'Sokoto GRU: Actual vs Predicted')
plot_residuals(sokoto_gru, sokoto_processed$x_test, sokoto_processed$y_test, 'Sokoto GRU: Residuals')

# Plot predictions vs actual and residuals for ENUGU LSTM
plot_predictions(enugu_lstm, enugu_processed$x_test, enugu_processed$y_test, 'Enugu LSTM: Actual vs Predicted')
plot_residuals(enugu_lstm, enugu_processed$x_test, enugu_processed$y_test, 'Enugu LSTM: Residuals')

# Plot predictions vs actual and residuals for ENUGU GRU
plot_predictions(enugu_gru, enugu_processed$x_test, enugu_processed$y_test, 'Enugu GRU: Actual vs Predicted')
plot_residuals(enugu_gru, enugu_processed$x_test, enugu_processed$y_test, 'Enugu GRU: Residuals')
