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

# Print summaries after each preprocessing step
print("Data after initial drop_na:")
print(summary(climatedata))
print("Data after mutating columns to numeric:")
print(summary(climatedata))
print("Data after replacing 0s with NA:")
print(summary(climatedata))
print("Data after final drop_na:")
print(summary(climatedata))
print("Data after log transformation:")
print(summary(climatedata))

# Create a list to store results for each city
city_results <- list()

# One-hot encode the 'city' column for overall dataset
climatedata <- climatedata %>%
  mutate(city = factor(city)) %>%
  bind_cols(model.matrix(~ city - 1, data = climatedata))

# Remove the original 'city' column
climatedata <- climatedata %>%
  select(-city)

# Print the structure of the dataset after encoding
print("Data after one-hot encoding 'city' column:")
print(str(climatedata))

# Get the unique cities
cities <- unique(climatedata$city)

# Define models
build_lstm_model <- function(input_shape, units = 50, learning_rate = 0.001, optimizer = 'adam', activation = 'tanh', batch_size = 32, epochs = 50) {
  input_layer <- layer_input(shape = input_shape)
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

build_gru_model <- function(input_shape, units = 50, learning_rate = 0.001, optimizer = 'adam', activation = 'tanh', batch_size = 32, epochs = 50) {
  input_layer <- layer_input(shape = input_shape)
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

hyper_grid <- expand.grid(
  units = c(50),
  learning_rate = c(0.0001, 0.001, 0.01, 0.1),
  batch_size = c(32, 64),  # Smaller batch size for faster execution
  epochs = c(20)           # Fewer epochs for debugging
)

# Continue train_and_evaluate_cv function
train_and_evaluate_cv <- function(build_model_func, hyper_grid, x_train, y_train, input_shape, folds = 5) {
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
      
      # Check for NA values and print shapes for debugging
      if (any(is.na(x_train_cv)) | any(is.na(y_train_cv)) | any(is.na(x_val_cv)) | any(is.na(y_val_cv))) {
        # Identify indices with NA values
        na_indices <- list(
          x_train_cv = which(is.na(x_train_cv), arr.ind = TRUE),
          y_train_cv = which(is.na(y_train_cv), arr.ind = TRUE),
          x_val_cv = which(is.na(x_val_cv), arr.ind = TRUE),
          y_val_cv = which(is.na(y_val_cv), arr.ind = TRUE)
        )
        print("NA values found in data:")
        print(na_indices)
        stop("NA values in data. Stopping execution.")
      }
      
      print(paste("x_train_cv shape:", dim(x_train_cv)))
      print(paste("y_train_cv shape:", dim(y_train_cv)))
      print(paste("x_val_cv shape:", dim(x_val_cv)))
      print(paste("y_val_cv shape:", dim(y_val_cv)))
      
      model_info <- build_model_func(input_shape, units = model_params$units, learning_rate = model_params$learning_rate, 
                                     batch_size = model_params$batch_size, epochs = model_params$epochs)
      model <- model_info$model
      history <- model %>% fit(
        x_train_cv, y_train_cv,
        epochs = model_info$epochs,
        batch_size = model_info$batch_size,
        validation_data = list(x_val_cv, y_val_cv),
        verbose = 1
      )
      
      list(loss = min(history$metrics$val_loss), mae = min(history$metrics$val_mean_absolute_error))
    }, simplify = FALSE)
    
    avg_loss <- mean(sapply(cv_results, function(x) x$loss))
    avg_mae <- mean(sapply(cv_results, function(x) x$mae))
    
    results[[params]] <- list(params = model_params, avg_loss = avg_loss, avg_mae = avg_mae)
  }
  return(results)
}


# Additional library for plotting
library(ggplot2)

# Function to plot training history
plot_history <- function(history, model_name) {
  metrics_df <- data.frame(
    epoch = 1:length(history$metrics$loss),
    loss = history$metrics$loss,
    val_loss = history$metrics$val_loss,
    mae = history$metrics$mean_absolute_error,
    val_mae = history$metrics$val_mean_absolute_error
  )
  
  ggplot(metrics_df, aes(x = epoch)) +
    geom_line(aes(y = loss, color = 'Training Loss')) +
    geom_line(aes(y = val_loss, color = 'Validation Loss')) +
    geom_line(aes(y = mae, color = 'Training MAE')) +
    geom_line(aes(y = val_mae, color = 'Validation MAE')) +
    labs(title = paste('Training History for', model_name),
         y = 'Metric Value', x = 'Epoch') +
    scale_color_manual(values = c('Training Loss' = 'blue', 'Validation Loss' = 'red', 'Training MAE' = 'green', 'Validation MAE' = 'purple')) +
    theme_minimal()
}

# Loop through each city
for (city in cities) {
  city_data <- climatedata %>%
    filter(city == city) %>%
    select(-matches("city"))
  
  # Split the data
  set.seed(123)
  train_indices <- sample(seq_len(nrow(city_data)), size = 0.8 * nrow(city_data))
  train_data <- city_data[train_indices, ]
  test_data <- city_data[-train_indices, ]
  
  # Select the relevant columns for features and targets
  x_train <- as.matrix(train_data %>% select(-sr, -rf, -rh))
  y_train <- as.matrix(train_data %>% select(sr, rf, rh))
  x_test <- as.matrix(test_data %>% select(-sr, -rf, -rh))
  y_test <- as.matrix(test_data %>% select(sr, rf, rh))
  
  # Reshape data for LSTM and GRU models
  x_train <- array_reshape(x_train, c(nrow(x_train), ncol(x_train), 1))
  x_test <- array_reshape(x_test, c(nrow(x_test), ncol(x_test), 1))
  
  # Print data shapes and summaries
  print(paste("City:", city))
  print("x_train shape:")
  print(dim(x_train))
  print("y_train shape:")
  print(dim(y_train))
  print("x_test shape:")
  print(dim(x_test))
  print("y_test shape:")
  print(dim(y_test))
  
  # Rebuild best LSTM model
  lstm_model_info <- build_lstm_model(dim(x_train)[2:3], units, learning_rate, 'adam', 'tanh', batch_size, epochs)
  lstm_model <- lstm_model_info$model
  
  # Fit the model on the entire training data
  lstm_history <- lstm_model %>% fit(
    x = x_train, y = y_train,
    epochs = epochs, batch_size = batch_size,
    validation_split = 0.2, verbose = 0
  )
  
  # Plot training history
  plot_history(lstm_history, "LSTM")
  
  # Evaluate the model on the test data
  lstm_evaluation <- lstm_model %>% evaluate(x_test, y_test, verbose = 0)
  
  # Save the best LSTM model
  save_model_hdf5(lstm_model, paste0("best_lstm_model_", city, ".h5"))
  
  # Store the results
  city_results[[city]] <- list(
    model_type = "LSTM",
    params = list(units = units, learning_rate = learning_rate, batch_size = batch_size, epochs = epochs),
    evaluation = lstm_evaluation,
    history = lstm_history
  )
  
  # Print the evaluation results
  print(paste("City:", city))
  print("Best LSTM model evaluation on test data:")
  print(lstm_evaluation)
  
  # Repeat the process for GRU model
  gru_results <- train_and_evaluate_cv(build_gru_model, hyper_grid, x_train, y_train, dim(x_train)[2:3])
  best_gru_result <- gru_results[which.min(gru_results$loss), ]
  
  # Extract parameters
  units <- best_gru_result$units
  learning_rate <- best_gru_result$learning_rate
  batch_size <- best_gru_result$batch_size
  epochs <- best_gru_result$epochs
  
  # Rebuild best GRU model
  gru_model_info <- build_gru_model(dim(x_train)[2:3], units, learning_rate, 'adam', 'tanh', batch_size, epochs)
  gru_model <- gru_model_info$model
  
  # Fit the model on the entire training data
  gru_history <- gru_model %>% fit(
    x = x_train, y = y_train,
    epochs = epochs, batch_size = batch_size,
    validation_split = 0.2, verbose = 0
  )
  
  # Plot training history
  plot_history(gru_history, "GRU")
  
  # Evaluate the model on the test data
  gru_evaluation <- gru_model %>% evaluate(x_test, y_test, verbose = 0)
  
  # Save the best GRU model
  save_model_hdf5(gru_model, paste0("best_gru_model_", city, ".h5"))
  
  # Store the results
  city_results[[city]] <- list(
    model_type = "GRU",
    params = list(units = units, learning_rate = learning_rate, batch_size = batch_size, epochs = epochs),
    evaluation = gru_evaluation,
    history = gru_history
  )
  
  # Print the evaluation results
  print(paste("City:", city))
  print("Best GRU model evaluation on test data:")
  print(gru_evaluation)
}

# Print the results for each city
print("Results for each city:")
print(city_results)
history <- model %>% fit(
  x = x_train_cv, y = y_train_cv,
  epochs = epochs, batch_size = batch_size,
  validation_data = list(x_val_cv, y_val_cv),
  verbose = 1
)

# Extract metrics from the last epoch
final_val_loss <- history$metrics$val_loss[epochs]
final_val_mae <- history$metrics$val_mean_absolute_error[epochs]

return(list(loss = final_val_loss, mae = final_val_mae))


avg_loss <- mean(sapply(cv_results, `[[`, "loss"))
avg_mae <- mean(sapply(cv_results, `[[`, "mae"))

results[[params]] <- c(model_params, avg_loss = avg_loss, avg_mae = avg_mae)


return(do.call(rbind, results))

# Split the data into features and target
x <- climatedata %>% select(-Year, -sr)
y <- climatedata$sr

# Normalize the features
x <- scale(x)

# Reshape the data for LSTM/GRU input
x <- array(x, dim = c(nrow(x), 1, ncol(x)))

# Train and evaluate the LSTM model using cross-validation
lstm_results <- train_and_evaluate_cv(
  build_lstm_model,
  hyper_grid,
  x,
  y,
  input_shape = c(1, ncol(x))
)

print("LSTM Results:")
print(lstm_results)

# Train and evaluate the GRU model using cross-validation
gru_results <- train_and_evaluate_cv(
  build_gru_model,
  hyper_grid,
  x,
  y,
  input_shape = c(1, ncol(x))
)

print("GRU Results:")
print(gru_results)

