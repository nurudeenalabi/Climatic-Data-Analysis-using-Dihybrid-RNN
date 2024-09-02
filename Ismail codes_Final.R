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
library(Metrics)  # For rmse
library(caret)  # For R2
# Set working directory
setwd("C:/Users/alabi/OneDrive/Desktop")
# Read the raw data from the CSV file
raw_climatedata <- read.csv("climatedatanew3.csv", stringsAsFactors = FALSE)
# Create a combined CityYear column for better x-axis representation
raw_climatedata <- raw_climatedata %>%
  mutate(CityYear = paste(city, year, sep = " "))
# Determine the positions where each city's data ends
city_breaks <- raw_climatedata %>%
  group_by(city) %>%
  summarize(end_position = n())

city_breaks <- cumsum(city_breaks$end_position)
# Plotting the time series data
ggplot(raw_climatedata, aes(x = 1:nrow(raw_climatedata), y = sr, color = city, group = city)) +
  geom_line() +
  labs(title = "Time Plot of Solar Radiation over Sokoto, Ilorin and Enugu ",
       x = "Period",
       y = "Solar Radiation ((ml)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  geom_vline(xintercept = city_breaks, liAnetype = "dashed", color = "red") +
  scale_color_manual(values = c("Sokoto" = "black", "Enugu" = "brown", "Ilorin" = "blue"))
# Load necessary libraries
library(tidyverse)
# Load the data
climatedata <- read.csv("climatedata_new.csv", header = TRUE, na.strings = "?")
# Select relevant columns
climatedata <- climatedata %>%
  select(year, city, sr, tmin, tmax, rh, sh, ep, ws, rf)
# Convert columns to numeric where necessary and handle NA values
climatedata <- climatedata %>%
  mutate(across(c(sr, tmin, tmax, rh, sh, ep, ws, rf), as.numeric)) %>%
  drop_na() %>%
  mutate(across(c(sr, tmin, tmax, rh, sh, ep, ws, rf), ~ifelse(. == 0, NA, .))) %>%
  drop_na() %>%
  mutate(across(c(sr, tmin, tmax, rh, sh, ep, ws, rf), log))
# Display the first few rows of the cleaned dataset
print(head(climatedata))
# Display the structure of the cleaned dataset
str(climatedata)
# Display a summary of the cleaned dataset
summary(climatedata)
# Ensure 'city' column is present in the dataset
if(!"city" %in% colnames(climatedata)) {
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

set.seed(123)
train_indices <- sample(seq_len(nrow(climatedata)), size = 0.8 * nrow(climatedata))
train_data <- climatedata[train_indices, ]
test_data <- climatedata[-train_indices, ]
# Select the relevant columns for features and target
x_train <- as.matrix(train_data %>% select(tmin, tmax, sh, ws, rh, ep, rf))
y_train <- as.matrix(train_data$sr)
x_test <- as.matrix(test_data %>% select(tmin, tmax, sh, ws, rh, ep, rf))
y_test <- as.matrix(test_data$sr)
# Reshape data for LSTM and GRU models
x_train <- array_reshape(x_train, c(nrow(x_train), ncol(x_train), 1))
x_test <- array_reshape(x_test, c(nrow(x_test), ncol(x_test), 1))
# Define custom RMSE metric
rmse <- function(y_true, y_pred) {
  K <- backend()
  return(K$sqrt(K$mean(K$square(y_pred - y_true), axis = as.integer(-1))))
}
# Define custom R-square metric
rsquare <- function(y_true, y_pred) {
  K <- backend()
  ss_res <- K$sum(K$square(y_true - y_pred))
  ss_tot <- K$sum(K$square(y_true - K$mean(y_true)))
  return(1 - ss_res / (ss_tot + K$epsilon()))
}
# Assuming seq_len is the sequence length, and num_features is the number of features
seq_len <- dim(x_train)[2]
num_features <- dim(x_train)[3]

# Function to create LSTM model
create_lstm_model <- function() {
  input_layer <- layer_input(shape = c(seq_len, num_features))
  lstm_layer_1 <- layer_lstm(units = 50, return_sequences = TRUE)(input_layer)
  dropout_layer_1 <- layer_dropout(rate = 0.6)(lstm_layer_1)
  lstm_layer_2 <- layer_lstm(units = 50)(dropout_layer_1)
  dropout_layer_2 <- layer_dropout(rate = 0.6)(lstm_layer_2)
  output_layer <- layer_dense(units = 1)(dropout_layer_2)
  model <- keras_model(inputs = input_layer, outputs = output_layer)
  return(model)
}

# Function to create GRU model
create_gru_model <- function() {
  input_layer <- layer_input(shape = c(seq_len, num_features))
  gru_layer_1 <- layer_gru(units = 50, return_sequences = TRUE)(input_layer)
  dropout_layer_1 <- layer_dropout(rate = 0.6)(gru_layer_1)
  gru_layer_2 <- layer_gru(units = 50)(dropout_layer_1)
  dropout_layer_2 <- layer_dropout(rate = 0.6)(gru_layer_2)
  output_layer <- layer_dense(units = 1)(dropout_layer_2)
  model <- keras_model(inputs = input_layer, outputs = output_layer)
  return(model)
}

# Function to train and evaluate model
train_and_evaluate_model <- function(model_fn, optimizer, x_train, y_train, x_test, y_test, epochs = 150, batch_size = 16) {
  model <- model_fn()
  model %>% compile(
    loss = "mse",
    optimizer = optimizer,
    metrics = "mae"
  )
  history <- model %>% fit(
    x_train, y_train,
    epochs = epochs,
    batch_size = batch_size,
    validation_split = 0.2,
    verbose = 2
  )
  loss <- model %>% evaluate(x_test, y_test)
  predictions <- model %>% predict(x_test)
  return(list(loss = loss[[1]], mae = loss[[2]], predictions = predictions, history = history))
}

# Define the optimizers
optimizers <- list(
  sgd = optimizer_sgd(learning_rate = 0.01, momentum = 0.9),
  rmsprop = optimizer_rmsprop(learning_rate = 0.001),
  adagrad = optimizer_adagrad(learning_rate = 0.01),
  adadelta = optimizer_adadelta(learning_rate = 1.0),
  adam = optimizer_adam(learning_rate = 0.001),
  adamax = optimizer_adamax(learning_rate = 0.002),
  nadam = optimizer_nadam(learning_rate = 0.002)
)

# Results storage
results <- list()

# Training models
for (opt_name in names(optimizers)) {
  cat("Training LSTM with optimizer:", opt_name, "\n")
  results[[paste("LSTM", opt_name, sep = "_")]] <- train_and_evaluate_model(
    create_lstm_model, optimizers[[opt_name]], x_train, y_train, x_test, y_test
  )
  
  cat("Training GRU with optimizer:", opt_name, "\n")
  results[[paste("GRU", opt_name, sep = "_")]] <- train_and_evaluate_model(
    create_gru_model, optimizers[[opt_name]], x_train, y_train, x_test, y_test
  )
}

# After training with all optimizers, you can compare the results
results_df <- tibble(
  model_optimizer = names(results),
  loss = sapply(results, function(res) res$loss),
  mae = sapply(results, function(res) res$mae)
)
print(results_df)

# Plot the results
results_df %>%
  pivot_longer(cols = c(loss, mae), names_to = "metric", values_to = "value") %>%
  ggplot(aes(x = model_optimizer, y = value, fill = metric)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Loss and MAE for Different Optimizers", x = "Model and Optimizer", y = "Value")

# Build LSTM model function with fixed parameters with best optimzer
build_lstm_model <- function(units = 150, learning_rate = 0.00001, optimizer = 'nadam', activation = 'relu', recurrent_activation = 'gelu', batch_size = 2, epochs = 150) {
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
build_gru_model <- function(units = 150, learning_rate = 0.0001, optimizer = 'adam', activation = 'tanh', recurrent_activation = 'hard_tanh', batch_size = 2, epochs = 150){
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
lstm_history = best_lstm_model_info$model %>% fit(
  x = x_train, y = y_train,
  epochs = best_lstm_model_info$epochs, batch_size = best_lstm_model_info$batch_size,
  validation_split = 0.2, verbose = 2
)
lstm_eval <- best_lstm_model_info$model %>% evaluate(x_test, y_test, verbose = 2)
best_lstm_preds <- best_lstm_model_info$model %>% predict(x_test)
print(lstm_eval)
# Calculate additional metrics for the lstm model
library(Metrics)  # For rmse
library(caret)  # For R2
rmse_value_lstm <- rmse(exp(y_test), exp(best_lstm_preds))
r2_value_lstm <- R2(exp(best_lstm_preds), exp(y_test))
print(r2_value_lstm)
# print(best_lstm_preds)
# Train and evaluate GRU model
best_gru_model_info <- build_gru_model()
gru_history = best_gru_model_info$model %>% fit(
  x = x_train, y = y_train,
  epochs = best_gru_model_info$epochs, batch_size = best_gru_model_info$batch_size,
  validation_split = 0.2, verbose = 2
)
gru_eval <- best_gru_model_info$model %>% evaluate(x_test, y_test, verbose = 2)
best_gru_preds <- best_gru_model_info$model %>% predict(x_test)
print(gru_eval)
# Calculate additional metrics for the lstm model
rmse_value_gru <- rmse(exp(y_test), exp(best_gru_preds))
r2_value_gru <- R2(exp(best_gru_preds), exp(y_test))
print(r2_value_gru)
# print(best_gru_preds)
# Extracting metrics from LSTM history
lstm_df <- data.frame(
  epoch = 1:length(lstm_history$metrics$loss),
  loss = lstm_history$metrics$loss,
  val_loss = lstm_history$metrics$val_loss,
  mean_absolute_error = lstm_history$metrics$mean_absolute_error,
  val_mean_absolute_error = lstm_history$metrics$val_mean_absolute_error
)
# Extracting metrics from GRU history
gru_df <- data.frame(
  epoch = 1:length(gru_history$metrics$loss),
  loss = gru_history$metrics$loss,
  val_loss = gru_history$metrics$val_loss,
  mean_absolute_error = gru_history$metrics$mean_absolute_error,
  val_mean_absolute_error = gru_history$metrics$val_mean_absolute_error
)
# Define a white theme for plots
white_theme <- theme_minimal() + 
  theme(panel.background = element_rect(fill = "white", colour = "white"),
        plot.background = element_rect(fill = "white", colour = "white"))
# Plotting LSTM history
plot1 = ggplot(lstm_df, aes(x = epoch)) +
  geom_line(aes(y = loss, color = "loss")) +
  labs(title = "LSTM Training History", x = "Epochs", y = "Loss") +
  scale_color_manual(values = c("loss" = "blue")) +
  white_theme
# Plotting LSTM mean absolute error
plot2 = ggplot(lstm_df, aes(x = epoch)) +
  geom_line(aes(y = mean_absolute_error, color = "mean_absolute_error")) +
  labs(title = "LSTM Training History", x = "Epochs", y = "Mean Absolute Error") +
  scale_color_manual(values = c("mean_absolute_error" = "green")) +
  white_theme
# Plotting GRU history
plot3 = ggplot(gru_df, aes(x = epoch)) +
  geom_line(aes(y = loss, color = "loss")) +
  labs(title = "GRU Training History", x = "Epochs", y = "Loss") +
  scale_color_manual(values = c("loss" = "green")) +
  white_theme
# Plotting GRU mean absolute error
plot4 = ggplot(gru_df, aes(x = epoch)) +
  geom_line(aes(y = mean_absolute_error, color = "mean_absolute_error")) +
  labs(title = "GRU Training History", x = "Epochs", y = "Mean Absolute Error") +
  scale_color_manual(values = c("mean_absolute_error" = "blue")) +
  white_theme
# Combine all plots into a single frame
grid.arrange(plot1, plot2, plot3, plot4, ncol = 2, nrow = 2)
# Reverse log transformation
y_test_exp <- exp(y_test)
best_lstm_preds_exp <- exp(best_lstm_preds)
best_gru_preds_exp <- exp(best_gru_preds)
# print(best_lstm_preds_exp)
# print(best_gru_preds_exp)
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
# print(predictions_df_combined)
head(predictions_df_combined);tail(predictions_df_combined)
# Plotting
plot_df_lstm <- data.frame(Actual = y_test_exp, Predicted = best_lstm_preds_exp)
lstm_plot <- ggplot(plot_df_lstm, aes(x = seq_along(Actual))) +
  geom_line(aes(y = Actual, color = "Actual"), size = 0.08) +
  geom_line(aes(y = Predicted, color = "Predicted"), size = 0.08) +
  labs(title = "LSTM: Actual vs Predicted", x = "Index", y = "Values") +
  scale_color_manual(values = c("Actual" = "red", "Predicted" = "blue"))

plot_df_gru <- data.frame(Actual = y_test_exp, Predicted = best_gru_preds_exp)
gru_plot <- ggplot(plot_df_gru, aes(x = seq_along(Actual))) +
  geom_line(aes(y = Actual, color = "Actual"), size = 0.08) +
  geom_line(aes(y = Predicted, color = "Predicted"), size = 0.08) +
  labs(title = "GRU: Actual vs Predicted", x = "Index", y = "Values") +
  scale_color_manual(values = c("Actual" = "green", "Predicted" = "black"))
plot_grid(lstm_plot, gru_plot, nrow = 1, ncol = 2)
# Residuals for LSTM
residuals_lstm <- y_test_exp - best_lstm_preds_exp
residuals_df_lstm <- data.frame(Residuals = residuals_lstm)
# residuals_df_lstm
# Plotting LSTM residuals histogram without boundary lines
lstm_residual_hist <- ggplot(residuals_df_lstm, aes(x = Residuals)) +
  geom_histogram(bins = 15, fill = "purple", color = NA) +  # No boundary line
  labs(title = "LSTM: Residuals Histogram", x = "Residuals", y = "Frequency")
# Residuals for GRU
residuals_gru <- y_test_exp - best_gru_preds_exp
residuals_df_gru <- data.frame(Residuals = residuals_gru)
# residuals_df_gru
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
# print(residuals_df_combined)
head(residuals_df_combined);tail(residuals_df_combined)
# Residuals for LSTM
residuals_lstm <- y_test_exp - best_lstm_preds_exp
residuals_df_lstm <- data.frame(Index = seq_along(residuals_lstm), Residuals = residuals_lstm)
# Plotting LSTM residuals histogram without boundary lines
lstm_residual_hist <- ggplot(residuals_df_lstm, aes(x = Residuals)) +
  geom_histogram(bins = 15, fill = "purple", color = NA) +
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
  labs(title = "LSTM: Actual vs Predicted", x = "Index", y = "Values") +
  scale_color_manual(values = c("Actual" = "red", "Predicted" = "blue"))
# Prediction vs. Actual for GRU
plot_df_gru <- data.frame(Actual = y_test_exp, Predicted = best_gru_preds_exp)
gru_vs_actual_plot <- ggplot(plot_df_gru, aes(x = seq_along(Actual))) +
  geom_line(aes(y = Actual, color = "Actual")) +
  geom_line(aes(y = Predicted, color = "Predicted")) +
  labs(title = "GRU: Actual vs Predicted", x = "Index", y = "Values") +
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

### HYBRID MODEL
input_shape <- dim(x_train)[-1]

hybrid_model <- keras_model_sequential() %>%
  layer_lstm(units = 150, return_sequences = TRUE, input_shape = input_shape) %>%
  layer_gru(units = 150, return_sequences = FALSE) %>%
  layer_dense(units = 1)

hybrid_model %>% compile(
  optimizer = 'nadam',
  loss = 'mean_squared_error',
  metrics = 'mean_absolute_error', 
)
summary(hybrid_model)
history <- hybrid_model %>% fit(
  x_train, y_train,
  epochs = 150,
  optimizer = 'adam',
  batch_size = 2, 
  recurrent_activation = swish,
  validation_data = list(x_test, y_test),
  verbose = 2
)
loss <- hybrid_model %>% evaluate(x_test, y_test, verbose = 2)
cat('Test loss:', loss, '\n')
predictions <- hybrid_model %>% predict(x_test)
# Calculate additional metrics for the hybrid model
rmse_value_hybrid <- rmse(exp(y_test), exp(predictions))
r2_value_hybrid <- R2(exp(predictions), exp(y_test))
print(r2_value_hybrid)
# Update the evaluation metrics for the hybrid model to include RMSE and R2
# hybrid_eval <- c(hybrid_eval, RMSE = rmse_value, R2 = r2_value)
# print(hybrid_eval)
# Convert the predictions and actual values to a data frame
results <- data.frame(
  Index = 1:length(y_test),
  Actual = exp(y_test),  # Reverse the log transformation
  Predicted = exp(predictions)  # Reverse the log transformation
)
# Line plot for actual vs predicted values
p1 <- ggplot(results, aes(x = Index)) +
  geom_line(aes(y = Actual, colour = "Actual"), size = 0.09) +
  geom_line(aes(y = Predicted, colour = "Predicted"), size = 0.09) +
  labs(title = "Actual vs Predicted Values",
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
# print(exp(predictions))
# print(Actual)
# Convert residuals to a data frame
residuals_df <- data.frame(
  Residuals = residuals
)
# Histogram of residuals
p2 <- ggplot(residuals_df, aes(x = Residuals)) +
  geom_histogram(fill = "purple", bins = 30, alpha = 0.7) +
  labs(title = "Residuals Histogram",
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
  Actual = y_test_exp,Predicted_hybrid = exp(predictions),
  Predicted_GRU = best_gru_preds_exp,Predicted_LSTM = best_lstm_preds_exp
  
)
# print(predictions_df_combined)
head(predictions_df_combined);tail(predictions_df_combined)
# Extract training history
dihybrid_history <- history
# Convert to data frame for plotting
dihybrid_df <- data.frame(
  epoch = 1:length(dihybrid_history$metrics$loss),
  loss = unlist(dihybrid_history$metrics$loss),
  val_loss = unlist(dihybrid_history$metrics$val_loss),
  mean_absolute_error = unlist(dihybrid_history$metrics$mean_absolute_error),
  val_mean_absolute_error = unlist(dihybrid_history$metrics$val_mean_absolute_error)
)
# Combine all three dataframes into a single dataframe for comparison
combined_df <- rbind(
  transform(dihybrid_df, model = "Dihybrid")
)
# Plot combined history with a white background
library(ggplot2)
# Plot for loss and validation loss
loss_plot <- ggplot(combined_df, aes(x = epoch, y = loss, color = model)) +
  geom_line() +
  geom_line(aes(y = val_loss), linetype = "dashed") +
  labs(title = "Model Loss", x = "Epoch", y = "Loss") +
  theme_minimal() +
  theme(panel.background = element_rect(fill = "white", color = NA))
# Plot for mean absolute error and validation mean absolute error
mae_plot <- ggplot(combined_df, aes(x = epoch, y = mean_absolute_error, color = model)) +
  geom_line() +
  geom_line(aes(y = val_mean_absolute_error), linetype = "dashed") +
  labs(title = "Mean Absolute Error", x = "Epoch", y = "Mean Absolute Error") +
  theme_minimal() +
  theme(panel.background = element_rect(fill = "white", color = NA))
# Print the plots
print(loss_plot)
print(mae_plot)
plot_grid(loss_plot, mae_plot, nrow = 1, ncol = 2)

#PARALLEL LSTM_GRU HYBRID MODEL
# Define a Parallel LSTM-GRU model function
build_parallel_lstm_gru_model <- function(units = 150, optimizer = 'adam', learning_rate = 0.000001, activation = 'relu', recurrent_activation = 'gelu', batch_size = 2, epochs = 150) {
  input_layer <- layer_input(shape = c(ncol(x_train), 1))
  # LSTM branch
  lstm_branch <- layer_lstm(units = units, activation = activation, return_sequences = TRUE)(input_layer)
  lstm_branch <- layer_lstm(units = units, activation = activation)(lstm_branch)
  # GRU branch
  gru_branch <- layer_gru(units = units, activation = activation, return_sequences = TRUE)(input_layer)
  gru_branch <- layer_gru(units = units, activation = activation)(gru_branch)
  # Concatenate LSTM and GRU branches
  concatenated <- layer_concatenate(list(lstm_branch, gru_branch))
  # Output layer
  output_layer <- layer_dense(units = 1, activation = 'linear')(concatenated)
  # Define and compile the model
  model <- keras_model(inputs = input_layer, outputs = output_layer)
  optimizer <- optimizer_adam(learning_rate = learning_rate)
  model %>% compile(
    loss = 'mean_squared_error',
    optimizer = optimizer,
    metrics = list('mean_absolute_error', custom_metric("rmse", rmse), custom_metric("rsquare", rsquare))
  )
  
  return(list(model = model, batch_size = batch_size, epochs = epochs))
}
# Train and evaluate the parallel LSTM-GRU model
best_parallel_model_info <- build_parallel_lstm_gru_model()
parallel_history = best_parallel_model_info$model %>% fit(
  x = x_train, y = y_train,
  epochs = best_parallel_model_info$epochs, batch_size = best_parallel_model_info$batch_size,
  validation_split = 0.2, verbose = 2
)
parallel_eval <- best_parallel_model_info$model %>% evaluate(x_test, y_test, verbose = 2)
best_parallel_preds <- best_parallel_model_info$model %>% predict(x_test)
# Calculate additional metrics for the parallel LSTM-GRU model
rmse_value_parallel <- rmse(exp(y_test), exp(best_parallel_preds))
r2_value_parallel <- R2(exp(best_parallel_preds), exp(y_test))
print(r2_value_parallel)
# Reverse log transformation
best_parallel_preds_exp <- exp(best_parallel_preds)
# Combine predictions into a single data frame
predictions_df_combined <- data.frame(
  Index = seq_along(y_test_exp),
  Actual = y_test_exp,
  Predicted_LSTM_GRU_Parallel = best_parallel_preds_exp
)
# Plotting
plot_df_parallel <- data.frame(Actual = y_test_exp, Predicted = best_parallel_preds_exp)
parallel_plot <- ggplot(plot_df_parallel, aes(x = seq_along(Actual))) +
  geom_line(aes(y = Actual, color = "Actual"), size = 0.08) +
  geom_line(aes(y = Predicted, color = "Predicted"), size = 0.08) +
  labs(title = "Parallel LSTM-GRU: Actual vs Predicted", x = "Index", y = "Values") +
  scale_color_manual(values = c("Actual" = "red", "Predicted" = "blue"))
# Display the plot
print(parallel_plot)

