# Load necessary libraries
library(tidyverse)
library(lubridate)
library(zoo)
library(reticulate)
library(tensorflow)
library(keras)

# Set working directory
setwd("C:/Users/alabi/OneDrive/Desktop")

# Load the data
climatedata <- read.csv("climatedata.csv", header = TRUE, na.strings = "?")

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

# Split the data into training and testing sets
set.seed(123)
train_indices <- sample(seq_len(nrow(climatedata)), size = 0.8 * nrow(climatedata))
train_data <- climatedata[train_indices, ]
test_data <- climatedata[-train_indices, ]

# Prepare the data for the models
x_train <- as.matrix(train_data %>% select(tmin, tmax, sh, ws, rh, ep, rf))
y_train <- as.matrix(train_data$sr)
x_test <- as.matrix(test_data %>% select(tmin, tmax, sh, ws, rh, ep, rf))
y_test <- as.matrix(test_data$sr)

# Reshape data for LSTM and GRU models
x_train <- array_reshape(x_train, c(nrow(x_train), ncol(x_train), 1))
x_test <- array_reshape(x_test, c(nrow(x_test), ncol(x_test), 1))

# Build LSTM model
build_lstm_model <- function(optimizer = 'adam', activation = 'tanh') {
  input_layer <- layer_input(shape = c(ncol(x_train), 1))
  lstm_layer <- layer_lstm(units = 50, activation = activation, return_sequences = TRUE)(input_layer)
  lstm_layer <- layer_lstm(units = 50, activation = activation)(lstm_layer)
  output_layer <- layer_dense(units = 1, activation = 'linear')(lstm_layer)
  
  model <- keras_model(inputs = input_layer, outputs = output_layer)
  model %>% compile(
    loss = 'mean_squared_error',
    optimizer = optimizer,
    metrics = list('mean_absolute_error')
  )
  return(model)
}

# Build GRU model
build_gru_model <- function(optimizer = 'adam', activation = 'tanh') {
  input_layer <- layer_input(shape = c(ncol(x_train), 1))
  gru_layer <- layer_gru(units = 50, activation = activation, return_sequences = TRUE)(input_layer)
  gru_layer <- layer_gru(units = 50, activation = activation)(gru_layer)
  output_layer <- layer_dense(units = 1, activation = 'linear')(gru_layer)
  
  model <- keras_model(inputs = input_layer, outputs = output_layer)
  model %>% compile(
    loss = 'mean_squared_error',
    optimizer = optimizer,
    metrics = list('mean_absolute_error')
  )
  return(model)
}

# Define function to train and evaluate models
train_and_evaluate <- function(model, x_train, y_train, x_test, y_test, epochs = 100, batch_size = 32) {
  history <- model %>% fit(
    x = x_train, y = y_train,
    epochs = epochs, batch_size = batch_size,
    validation_split = 0.2, verbose = 0
  )
  evaluation <- model %>% evaluate(x_test, y_test)
  predictions <- model %>% predict(x_test)
  return(list(history = history, evaluation = evaluation, predictions = predictions))
}

# Optimizers and activation functions to compare
optimizers <- c('adam', 'rmsprop', 'sgd')
activations <- c('tanh', 'relu', 'sigmoid')

# Train and evaluate LSTM models with different optimizers and activation functions
lstm_results <- expand.grid(optimizers = optimizers, activations = activations) %>%
  rowwise() %>%
  do({
    model <- build_lstm_model(.$optimizers, .$activations)
    result <- train_and_evaluate(model, x_train, y_train, x_test, y_test)
    data.frame(optimizer = .$optimizers, activation = .$activations, loss = result$evaluation[[1]], mae = result$evaluation[[2]], model = I(list(result)))
  })

# Train and evaluate GRU models with different optimizers and activation functions
gru_results <- expand.grid(optimizers = optimizers, activations = activations) %>%
  rowwise() %>%
  do({
    model <- build_gru_model(.$optimizers, .$activations)
    result <- train_and_evaluate(model, x_train, y_train, x_test, y_test)
    data.frame(optimizer = .$optimizers, activation = .$activations, loss = result$evaluation[[1]], mae = result$evaluation[[2]], model = I(list(result)))
  })

# Find the best LSTM and GRU models based on the lowest loss
best_lstm_result <- lstm_results %>% filter(loss == min(loss))
best_gru_result <- gru_results %>% filter(loss == min(loss))

# Extract predictions from the best models
pred_lstm <- best_lstm_result$model[[1]]$predictions
pred_gru <- best_gru_result$model[[1]]$predictions

# Plot actual vs predicted values for the best LSTM model
plot_df <- data.frame(Actual = y_test, Predicted = pred_lstm)
ggplot(plot_df, aes(x = seq_along(Actual))) +
  geom_line(aes(y = Actual, color = "Actual")) +
  geom_line(aes(y = Predicted, color = "Predicted")) +
  labs(title = "LSTM: Actual vs Predicted", x = "Index", y = "Values") +
  scale_color_manual(values = c("Actual" = "blue", "Predicted" = "red"))

# Plot actual vs predicted values for the best GRU model
plot_df <- data.frame(Actual = y_test, Predicted = pred_gru)
ggplot(plot_df, aes(x = seq_along(Actual))) +
  geom_line(aes(y = Actual, color = "Actual")) +
  geom_line(aes(y = Predicted, color = "Predicted")) +
  labs(title = "GRU: Actual vs Predicted", x = "Index", y = "Values") +
  scale_color_manual(values = c("Actual" = "black", "Predicted" = "green"))

# Report results for each city
results_by_city <- climatedata %>%
  group_by(city) %>%
  group_modify(~ {
    train_indices <- sample(seq_len(nrow(.x)), size = 0.8 * nrow(.x))
    train_data <- .x[train_indices, ]
    test_data <- .x[-train_indices, ]
    
    x_train <- as.matrix(train_data %>% select(tmin, tmax, sh, ws, rh, ep, rf))
    y_train <- as.matrix(train_data$sr)
    x_test <- as.matrix(test_data %>% select(tmin, tmax, sh, ws, rh, ep, rf))
    y_test <- as.matrix(test_data$sr)
    
    x_train <- array_reshape(x_train, c(nrow(x_train), ncol(x_train), 1))
    x_test <- array_reshape(x_test, c(nrow(x_test), ncol(x_test), 1))
    
    lstm_model <- build_lstm_model('adam', 'tanh')
    gru_model <- build_gru_model('adam', 'tanh')
    
    lstm_result <- train_and_evaluate(lstm_model, x_train, y_train, x_test, y_test)
    gru_result <- train_and_evaluate(gru_model, x_train, y_train, x_test, y_test)
    
    data.frame(
      city = unique(.x$city), 
      lstm_mae = lstm_result$evaluation[[2]], 
      gru_mae = gru_result$evaluation[[2]]
    )
  })

print(results_by_city)
