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
build_lstm_model <- function(units = 100, learning_rate = 0.1, optimizer = 'adam', activation = 'relu', batch_size = 64, epochs = 100) {
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
build_gru_model <- function(units = 100, learning_rate = 0.01, optimizer = 'adam', activation = 'relu', batch_size = 64, epochs = 100) {
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
lstm_eval <- best_lstm_model_info$model %>% evaluate(x_test, y_test, verbose = 0)
best_lstm_preds <- best_lstm_model_info$model %>% predict(x_test)

# Train and evaluate GRU model
best_gru_model_info <- build_gru_model()
best_gru_model_info$model %>% fit(
  x = x_train, y = y_train,
  epochs = best_gru_model_info$epochs, batch_size = best_gru_model_info$batch_size,
  validation_split = 0.2, verbose = 0
)
gru_eval <- best_gru_model_info$model %>% evaluate(x_test, y_test, verbose = 0)
best_gru_preds <- best_gru_model_info$model %>% predict(x_test)

# Reverse log transformation
y_test_exp <- exp(y_test)
best_lstm_preds_exp <- exp(best_lstm_preds)
best_gru_preds_exp <- exp(best_gru_preds)

# Plotting
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

# Report results for each city
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
    
    k <- 5  # Number of cross-validation folds
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
