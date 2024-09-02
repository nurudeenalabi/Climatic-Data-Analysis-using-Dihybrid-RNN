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
require(ggplot2)
library(rBayesianOptimization)
library(forecast)
# Set working directory
setwd("C:/Users/alabi/OneDrive/Desktop")
# Load the data
climatedata <- read.csv("climatedatanew.csv", header = TRUE, na.strings = "?")
climatedata <- climatedata %>%
  select(year, city, sr, tmin, tmax, rh, sh, ep, ws, rf) %>%
  mutate(across(c(sr, tmin, tmax, rh, sh, ep, ws, rf), as.numeric)) %>%
  drop_na() %>%
  mutate(across(c(sr, tmin, tmax, rh, sh, ep, ws, rf), ~ifelse(. == 0, NA, .))) %>%
  drop_na() %>%
  mutate(across(c(sr, tmin, tmax, rh, sh, ep, ws, rf), log))

# Ensure 'city' column is present in the dataset
if (!"city" %in% colnames(climatedata)) {
  stop("Column 'city' is not found in the dataset.")
}

# Function to decompose the time series and add components as features
decompose_and_add_features <- function(data, target_var, freq) {
  decomposed <- stl(ts(data[[target_var]], frequency = freq), s.window = "periodic")
  data <- data %>%
    mutate(trend = as.numeric(decomposed$time.series[, "trend"]),
           seasonal = as.numeric(decomposed$time.series[, "seasonal"]),
           remainder = as.numeric(decomposed$time.series[, "remainder"]))
  return(data)
}

# Decompose and add features
climatedata <- decompose_and_add_features(climatedata, "sr", freq = 365)

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

x_train <- as.matrix(train_data %>% select(tmin, tmax, sh, ws, rh, ep, rf, trend, seasonal, remainder))
y_train <- as.matrix(train_data$sr)
x_test <- as.matrix(test_data %>% select(tmin, tmax, sh, ws, rh, ep, rf, trend, seasonal, remainder))
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
  validation_split = 0.2, verbose = 2
)
lstm_eval <- best_lstm_model_info$model %>% evaluate(x_test, y_test, verbose = 2)
best_lstm_preds <- best_lstm_model_info$model %>% predict(x_test)
print(lstm_eval)
print(best_lstm_preds)

# Train and evaluate GRU model
best_gru_model_info <- build_gru_model()
best_gru_model_info$model %>% fit(
  x = x_train, y = y_train,
  epochs = best_gru_model_info$epochs, batch_size = best_gru_model_info$batch_size,
  validation_split = 0.2, verbose = 2
)
gru_eval <- best_gru_model_info$model %>% evaluate(x_test, y_test, verbose = 2)
best_gru_preds <- best_gru_model_info$model %>% predict(x_test)
print(gru_eval)
print(best_gru_preds)

# Reverse log transformation
y_test_exp <- exp(y_test)
best_lstm_preds_exp <- exp(best_lstm_preds)
best_gru_preds_exp <- exp(best_gru_preds)
print(best_lstm_preds_exp)
print(best_gru_preds_exp)

summary(best_lstm_model_info$model)
summary(best_gru_model_info$model)

# Combine predictions into a single data frame
predictions_df_combined <- data.frame(
  Index = seq_along(y_test_exp),
  Actual = y_test_exp,
  Predicted_LSTM = best_lstm_preds_exp,
  Predicted_GRU = best_gru_preds_exp
)

# Display the combined data frame
print(predictions_df_combined)

# Plotting
plot_df_lstm <- data.frame(Actual = y_test_exp, Predicted = best_lstm_preds_exp)
lstm_plot <- ggplot(plot_df_lstm, aes(x = seq_along(Actual))) +
  geom_line(aes(y = Actual, color = "Actual")) +
  geom_line(aes(y = Predicted, color = "Predicted")) +
  labs(title = "LSTM: Actual vs Predicted on Solar Radiation", x = "Index", y = "Values") +
  scale_color_manual(values = c("Actual" = "red", "Predicted" = "blue"))

plot_df_gru <- data.frame(Actual = y_test_exp, Predicted = best_gru_preds_exp)
gru_plot <- ggplot(plot_df_gru, aes(x = seq_along(Actual))) +
  geom_line(aes(y = Actual, color = "Actual")) +
  geom_line(aes(y = Predicted, color = "Predicted")) +
  labs(title = "GRU: Actual vs Predicted on Solar Radiation", x = "Index", y = "Values") +
  scale_color_manual(values = c("Actual" = "green", "Predicted" = "black"))

plot_grid(lstm_plot, gru_plot, nrow = 1, ncol = 2)

# Residuals for LSTM
residuals_lstm <- y_test_exp - best_lstm_preds_exp
residuals_df_lstm <- data.frame(Residuals = residuals_lstm)
residuals_df_lstm
# Plotting LSTM residuals histogram without boundary lines
lstm_residual_hist <- ggplot(residuals_df_lstm, aes(x = Residuals)) +
  geom_histogram(bins = 30, fill = "purple", color = NA) +  # No boundary line
  labs(title = "LSTM: Residuals Histogram", x = "Residuals", y = "Frequency")

# Residuals for GRU
residuals_gru <- y_test_exp - best_gru_preds_exp
residuals_df_gru <- data.frame(Residuals = residuals_gru)
residuals_df_gru
# Plotting GRU residuals histogram without boundary lines
gru_residual_hist <- ggplot(residuals_df_gru, aes(x = Residuals)) +
  geom_histogram(bins = 30, fill = "green", color = NA) +  # No boundary line
  labs(title = "GRU: Residuals Histogram", x = "Residuals", y = "Frequency")

# Combine the plots using plot_grid
combined_residual_plots <- plot_grid(
  lstm_residual_hist + theme(plot.title = element_text(hjust = 0.5)),
  gru_residual_hist + theme(plot.title = element_text(hjust = 0.5)),
  labels = c("A", "B"),
  ncol = 2
)

# Display the combined plot
print(combined_residual_plots)

# Residuals for LSTM
residuals_lstm <- y_test_exp - best_lstm_preds_exp

# Residuals for GRU
residuals_gru <- y_test_exp - best_gru_preds_exp

# Combine into a single data frame
residuals_df_combined <- data.frame(
  Index = seq_along(residuals_lstm),
  Residuals_LSTM = residuals_lstm,
  Residuals_GRU = residuals_gru
)

# Display the combined data frame
print(residuals_df_combined)

# Residuals for LSTM
residuals_lstm <- y_test_exp - best_lstm_preds_exp
residuals_df_lstm <- data.frame(Index = seq_along(residuals_lstm), Residuals = residuals_lstm)

# Plotting LSTM residuals histogram without boundary lines
lstm_residual_hist <- ggplot(residuals_df_lstm, aes(x = Residuals)) +
  geom_histogram(bins = 30, fill = "purple", color = NA) +
  labs(title = "LSTM: Residuals Histogram", x = "Residuals", y = "Frequency")

# Residuals for GRU
residuals_gru <- y_test_exp - best_gru_preds_exp
residuals_df_gru <- data.frame(Index = seq_along(residuals_gru), Residuals = residuals_gru)

# Plotting GRU residuals histogram without boundary lines
gru_residual_hist <- ggplot(residuals_df_gru, aes(x = Residuals)) +
  geom_histogram(bins = 30, fill = "green", color = NA) +
  labs(title = "GRU: Residuals Histogram", x = "Residuals", y = "Frequency")

# Prediction vs. Actual for LSTM
plot_df_lstm <- data.frame(Actual = y_test_exp, Predicted = best_lstm_preds_exp)
lstm_vs_actual_plot <- ggplot(plot_df_lstm, aes(x = seq_along(Actual))) +
  geom_line(aes(y = Actual, color = "Actual")) +
  geom_line(aes(y = Predicted, color = "Predicted")) +
  labs(title = "LSTM: Actual vs Predicted on Solar Radiation", x = "Index", y = "Values") +
  scale_color_manual(values = c("Actual" = "red", "Predicted" = "blue"))

# Prediction vs. Actual for GRU
plot_df_gru <- data.frame(Actual = y_test_exp, Predicted = best_gru_preds_exp)
gru_vs_actual_plot <- ggplot(plot_df_gru, aes(x = seq_along(Actual))) +
  geom_line(aes(y = Actual, color = "Actual")) +
  geom_line(aes(y = Predicted, color = "Predicted")) +
  labs(title = "GRU: Actual vs Predicted on Solar Radiation", x = "Index", y = "Values") +
  scale_color_manual(values = c("Actual" = "green", "Predicted" = "black"))

# Combine all plots into a grid
combined_plots <- plot_grid(
  lstm_vs_actual_plot + theme(plot.title = element_text(hjust = 0.5)),
  gru_vs_actual_plot + theme(plot.title = element_text(hjust = 0.5)),
  lstm_residual_hist + theme(plot.title = element_text(hjust = 0.5)),
  gru_residual_hist + theme(plot.title = element_text(hjust = 0.5)),
  labels = c("A", "B", "C", "D"),
  ncol = 2
)

# Display the combined grid of plots
print(combined_plots)

# ###HYBRID MODEL
# input_shape <- dim(x_train)[-1]
# 
# hybrid_model <- keras_model_sequential() %>%
#   layer_lstm(units = 50, return_sequences = TRUE, input_shape = input_shape) %>%
#   layer_gru(units = 50, return_sequences = FALSE) %>%
#   layer_dense(units = 1)
# 
# hybrid_model %>% compile(
#   optimizer = 'adam',
#   loss = 'mean_squared_error',
#   metrics = 'mean_absolute_error'
# )
# 
# summary(hybrid_model)
# history <- hybrid_model %>% fit(
#   x_train, y_train,
#   epochs = 200,
#   batch_size = 8,
#   recurrent_activation = 'hard_tanh',
#   validation_data = list(x_test, y_test),
#   verbose = 2
# )
# loss <- hybrid_model %>% evaluate(x_test, y_test, verbose = 0)
# cat('Test loss:', loss, '\n')
# predictions <- hybrid_model %>% predict(x_test)
# # Sample evaluation metrics for demonstration (replace with your actual metrics)
# lstm_eval <- c(Loss = 0.05262252, MAE = 0.16555671)
# gru_eval <- c(Loss = 0.05917004, MAE = 0.18611154)
# hybrid_eval <- c(Loss =  0.0003402008, MAE = 0.01535103)
# 
# # Combine into a single data frame
# eval_df <- data.frame(
#   Model = c("LSTM", "GRU", "Hybrid"),
#   Loss = c(lstm_eval["Loss"], gru_eval["Loss"], hybrid_eval["Loss"]),
#   MAE = c(lstm_eval["MAE"], gru_eval["MAE"], hybrid_eval["MAE"])
# )
# 
# # Display the combined evaluation metrics table
# print(eval_df)
# 
# # Convert the predictions and actual values to a data frame
# results <- data.frame(
#   Index = 1:length(y_test),
#   Actual = y_test,
#   Predicted = predictions
# )
# 
# # Line plot for actual vs predicted values
# p1 <- ggplot(results, aes(x = Index)) +
#   geom_line(aes(y = Actual, colour = "Actual"), size = 1) +
#   geom_line(aes(y = Predicted, colour = "Predicted"), size = 1) +
#   labs(title = "Actual vs Predicted Values on Solar Radiation",
#        x = "Index",
#        y = "Values",
#        colour = "Legend") +
#   theme_minimal()
# 
# # Print the plot
# print(p1)
# # Calculate residuals
# residuals <- y_test - predictions
# 
# # Convert residuals to a data frame
# residuals_df <- data.frame(
#   Residuals = residuals
# )
# 
# # Histogram of residuals
# p2 <- ggplot(residuals_df, aes(x = Residuals)) +
#   geom_histogram(fill = "purple", bins = 30, alpha = 0.7) +
#   labs(title = "Residuals Histogram",
#        x = "Residuals",
#        y = "Frequency") +
#   theme_minimal()
# 
# # Print the plot
# print(p2)
# # install.packages("gridExtra")
# library(gridExtra)
# 
# # Combine plots
# grid.arrange(p1, p2, ncol = 2)
# 
# 
# set.seed(123)
# y_test <- y_test
# predictions <- hybrid_model %>% predict(x_test)
# print(predictions)
# # Create data frames for visualization
# results <- data.frame(
#   Index = 1:length(y_test),
#   Actual = y_test,
#   Predicted = predictions
# )
# 
# residuals_df <- data.frame(
#   Residuals = y_test - predictions
# )
# 
# # Plot actual vs predicted values
# p1 <- ggplot(results, aes(x = Index)) +
#   geom_line(aes(y = Actual, colour = "Actual"), size = 0.8) +
#   geom_line(aes(y = Predicted, colour = "Predicted"), size = 0.8) +
#   labs(title = "Actual vs Predicted Values on Solar Radiation",
#        x = "Index",
#        y = "Values",
#        colour = "Legend") +
#   theme_minimal() +
#   theme(
#     plot.title = element_text(size = 10, face = "bold"),
#     axis.title = element_text(size = 12),
#     legend.position = "bottom"
#   )
# 
# # Plot residuals histogram
# p2 <- ggplot(residuals_df, aes(x = Residuals)) +
#   geom_histogram(fill = "purple", bins = 30, alpha = 0.7) +
#   labs(title = "Residuals Histogram",
#        x = "Residuals",
#        y = "Frequency") +
#   theme_minimal() +
#   theme(
#     plot.title = element_text(size = 10, face = "bold"),
#     axis.title = element_text(size = 12)
#   )
# 
# # Combine plots
# grid.arrange(p1, p2, ncol = 2)


# Required libraries

library(Metrics)  # For rmse
library(caret)  # For R2

### HYBRID MODEL
input_shape <- dim(x_train)[-1]

hybrid_model <- keras_model_sequential() %>%
  layer_lstm(units = 50, return_sequences = TRUE, input_shape = input_shape) %>%
  layer_gru(units = 50, return_sequences = FALSE) %>%
  layer_dense(units = 1)

hybrid_model %>% compile(
  optimizer = 'adam',
  loss = 'mean_squared_error',
  metrics = 'mean_absolute_error'
)

summary(hybrid_model)
history <- hybrid_model %>% fit(
  x_train, y_train,
  epochs = 100,
  batch_size = 16, recurrent_activation = 'hard_tanh',
  validation_data = list(x_test, y_test),
  verbose = 2
)
loss <- hybrid_model %>% evaluate(x_test, y_test, verbose = 2)
cat('Test loss:', loss, '\n')
predictions <- hybrid_model %>% predict(x_test)

# Sample evaluation metrics for demonstration (replace with your actual metrics)
lstm_eval <- c(Loss = 0.05262252, MAE = 0.16555671)
gru_eval <- c(Loss = 0.05917004, MAE = 0.18611154)
hybrid_eval <- c(Loss = 0.0003402008, MAE = 0.01535103)

# Calculate additional metrics for the hybrid model
rmse_value <- rmse(exp(y_test), exp(predictions))
r2_value <- R2(exp(predictions), exp(y_test))
print(r2_value)
# Update the evaluation metrics for the hybrid model to include RMSE and R2
hybrid_eval <- c(hybrid_eval, RMSE = rmse_value, R2 = r2_value)

# # Combine into a single data frame
# eval_df <- data.frame(
#   Model = c("LSTM", "GRU", "Hybrid"),
#   Loss = c(lstm_eval["Loss"], gru_eval["Loss"], hybrid_eval["Loss"]),
#   MAE = c(lstm_eval["MAE"], gru_eval["MAE"], hybrid_eval["MAE"]),
#   RMSE = c(NA, NA, hybrid_eval["RMSE"]),  # NA for LSTM and GRU as they're not calculated
#   R2 = c(NA, NA, hybrid_eval["R2"])  # NA for LSTM and GRU as they're not calculated
# )

# Display the combined evaluation metrics table
# print(eval_df)

# Convert the predictions and actual values to a data frame
results <- data.frame(
  Index = 1:length(y_test),
  Actual = exp(y_test),  # Reverse the log transformation
  Predicted = exp(predictions)  # Reverse the log transformation
)

# Line plot for actual vs predicted values
p1 <- ggplot(results, aes(x = Index)) +
  geom_line(aes(y = Actual, colour = "Actual"), size = 1) +
  geom_line(aes(y = Predicted, colour = "Predicted"), size = 1) +
  labs(title = "Actual vs Predicted Values on Solar Radiation(Original Scale)",
       x = "Index",
       y = "Values",
       colour = "Legend") +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 10, face = "bold"),
    axis.title = element_text(size = 12),
    legend.position = "bottom"
  )

# Calculate residuals
residuals <- exp(y_test) - exp(predictions)  # Reverse the log transformation
print(exp(predictions))
print(Actual)
# Convert residuals to a data frame
residuals_df <- data.frame(
  Residuals = residuals
)

# Histogram of residuals
p2 <- ggplot(residuals_df, aes(x = Residuals)) +
  geom_histogram(fill = "purple", bins = 30, alpha = 0.7) +
  labs(title = "Residuals Histogram (Original Scale)",
       x = "Residuals",
       y = "Frequency") +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 10, face = "bold"),
    axis.title = element_text(size = 12)
  )

# Combine plots
grid.arrange(p1, p2, ncol = 2)
# Combine predictions into a single data frame
predictions_df_combined <- data.frame(
  Index = seq_along(y_test_exp),
  Actual = y_test_exp,
  Predicted_LSTM = best_lstm_preds_exp,
  Predicted_GRU = best_gru_preds_exp,
  Predicted_hybrid = exp(predictions)
)

print(predictions_df_combined)

