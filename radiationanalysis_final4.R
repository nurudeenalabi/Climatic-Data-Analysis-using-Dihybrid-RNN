# Load necessary libraries
library(tidyverse)
library(lubridate)
library(zoo)
library(reticulate)
library(tensorflow)
library(keras)
library(caret)

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

# Prepare the data for the models
x_data <- as.matrix(climatedata %>% select(tmin, tmax, sh, ws, rh, ep, rf))
y_data <- as.matrix(climatedata$sr)

# Normalize the data
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

x_data <- normalize(x_data)

# Reshape data for LSTM and GRU models
x_data <- array_reshape(x_data, c(nrow(x_data), ncol(x_data), 1))

# Hyperparameter grid with increased batch size
hyper_grid <- expand.grid(
  units = 100,
  learning_rate = c(0.0001, 0.01),
  batch_size = c(64),  # Increased batch size
  epochs = 50
)

# Function to build LSTM model
build_lstm_model <- function(units, learning_rate) {
  input_layer <- layer_input(shape = c(ncol(x_data), 1))
  lstm_layer <- layer_lstm(units = units, activation = 'tanh', return_sequences = TRUE)(input_layer)
  lstm_layer <- layer_lstm(units = units, activation = 'tanh')(lstm_layer)
  output_layer <- layer_dense(units = 1, activation = 'linear')(lstm_layer)
  
  model <- keras_model(inputs = input_layer, outputs = output_layer)
  optimizer <- optimizer_adam(learning_rate = learning_rate)
  model %>% compile(
    loss = 'mean_squared_error',
    optimizer = optimizer,
    metrics = list('mean_absolute_error')
  )
  return(model)
}

# Function to build GRU model
build_gru_model <- function(units, learning_rate) {
  input_layer <- layer_input(shape = c(ncol(x_data), 1))
  gru_layer <- layer_gru(units = units, activation = 'tanh', return_sequences = TRUE)(input_layer)
  gru_layer <- layer_gru(units = units, activation = 'tanh')(gru_layer)
  output_layer <- layer_dense(units = 1, activation = 'linear')(gru_layer)
  
  model <- keras_model(inputs = input_layer, outputs = output_layer)
  optimizer <- optimizer_adam(learning_rate = learning_rate)
  model %>% compile(
    loss = 'mean_squared_error',
    optimizer = optimizer,
    metrics = list('mean_absolute_error')
  )
  return(model)
}

# Function to train and evaluate models using cross-validation
train_and_evaluate_cv <- function(build_model_fn, hyper_grid, k = 5) {
  set.seed(123)
  folds <- createFolds(y_data, k = k, list = TRUE, returnTrain = TRUE)
  
  results <- list()
  
  for (i in 1:nrow(hyper_grid)) {
    hyperparams <- hyper_grid[i, ]
    cv_results <- data.frame()
    
    for (j in 1:k) {
      train_indices <- folds[[j]]
      test_indices <- setdiff(seq_len(nrow(x_data)), train_indices)
      
      x_train <- x_data[train_indices, , , drop = FALSE]
      y_train <- y_data[train_indices, , drop = FALSE]
      x_test <- x_data[test_indices, , , drop = FALSE]
      y_test <- y_data[test_indices, , drop = FALSE]
      
      model <- build_model_fn(hyperparams$units, hyperparams$learning_rate)
      
      model %>% fit(
        x = x_train, y = y_train,
        epochs = hyperparams$epochs, batch_size = hyperparams$batch_size,
        validation_split = 0.2, verbose = 1  # Reduce verbosity to 1 or 0
      )
      
      evaluation <- model %>% evaluate(x_test, y_test, verbose = 0)  # Reduce verbosity to 0
      cv_results <- rbind(cv_results, data.frame(loss = evaluation[[1]], mae = evaluation[[2]]))
    }
    
    avg_loss <- mean(cv_results$loss)
    avg_mae <- mean(cv_results$mae)
    
    results <- rbind(results, cbind(hyperparams, avg_loss, avg_mae))
  }
  
  return(results)
}

# Train and evaluate LSTM models
lstm_results <- train_and_evaluate_cv(build_lstm_model, hyper_grid)
summary(lstm_results)
# Train and evaluate GRU models
gru_results <- train_and_evaluate_cv(build_gru_model, hyper_grid)
summary(gru_results)
# Find the best LSTM and GRU models based on the lowest average loss
best_lstm_result <- lstm_results %>% filter(avg_loss == min(avg_loss))
best_gru_result <- gru_results %>% filter(avg_loss == min(avg_loss))
print(best_lstm_result)
print(best_gru_result)

# Extract the best hyperparameters for LSTM and GRU
best_lstm_model <- build_lstm_model(best_lstm_result$units, best_lstm_result$learning_rate)
best_gru_model <- build_gru_model(best_gru_result$units, best_gru_result$learning_rate)

# Train the best models on the entire dataset
best_lstm_model %>% fit(
  x = x_data, y = y_data,
  epochs = best_lstm_result$epochs, batch_size = best_lstm_result$batch_size,
  validation_split = 0.2, verbose = 1
)
best_gru_model %>% fit(
  x = x_data, y = y_data,
  epochs = best_gru_result$epochs, batch_size = best_gru_result$batch_size,
  validation_split = 0.2, verbose = 1
)

# Make predictions with the best models
pred_lstm <- best_lstm_model %>% predict(x_data)
pred_gru <- best_gru_model %>% predict(x_data)

# Plot actual vs predicted values for the best LSTM model
plot_df_lstm <- data.frame(Actual = y_data, Predicted = pred_lstm)
ggplot(plot_df_lstm, aes(x = seq_along(Actual))) +
  geom_line(aes(y = Actual, color = "Actual")) +
  geom_line(aes(y = Predicted, color = "Predicted")) +
  labs(title = "LSTM: Actual vs Predicted", x = "Index", y = "Values") +
  scale_color_manual(values = c("Actual" = "red", "Predicted" = "blue"))

# Plot actual vs predicted values for the best GRU model
plot_df_gru <- data.frame(Actual = y_data, Predicted = pred_gru)
ggplot(plot_df_gru, aes(x = seq_along(Actual))) +
  geom_line(aes(y = Actual, color = "Actual")) +
  geom_line(aes(y = Predicted, color = "Predicted")) +
  labs(title = "GRU: Actual vs Predicted", x = "Index", y = "Values") +
  scale_color_manual(values = c("Actual" = "green", "Predicted" = "black"))

# # Report results for each city with cross-validation
# results_by_city <- climatedata %>%
#   group_by(city) %>%
#   group_modify(~ {
#     if (nrow(.x) < 2) return(tibble(city = unique(.x$city), lstm_mae = NA, gru_mae = NA, lstm_pred = NA, gru_pred = NA))
#     
#     lstm_mae <- c()
#     gru_mae <- c()
#     lstm_preds <- c()
#     gru_preds <- c()
#     folds <- createFolds(.x$sr, k = 5, list = TRUE, returnTrain = TRUE)
#     
#     for (fold in folds) {
#       train_indices <- fold
#       test_indices <- setdiff(seq_len(nrow(.x)), train_indices)
#       
#       x_city <- as.matrix(.x %>% select(tmin, tmax, sh, ws, rh, ep, rf))
#       y_city <- as.matrix(.x$sr)
#       
#       x_city <- array_reshape(x_city, c(nrow(x_city), ncol(x_city), 1))
#       
#       x_train <- x_city[train_indices, , , drop = FALSE]
#       y_train <- y_city[train_indices, , drop = FALSE]
#       x_test <- x_city[test_indices, , , drop = FALSE]
#       y_test <- y_city[test_indices, , drop = FALSE]
#       
#       best_lstm_model %>% fit(
#         x = x_train, y = y_train,
#         epochs = best_lstm_result$epochs, batch_size = best_lstm_result$batch_size,
#         validation_split = 0.2, verbose = 0
#       )
#       lstm_eval <- best_lstm_model %>% evaluate(x_test, y_test, verbose = 0)
#       lstm_mae <- c(lstm_mae, lstm_eval[[2]])
#       
#       best_lstm_preds <- best_lstm_model %>% predict(x_test)
#       lstm_preds <- c(lstm_preds, best_lstm_preds)
#       
#       best_gru_model %>% fit(
#         x = x_train, y = y_train,
#         epochs = best_gru_result$epochs, batch_size = best_gru_result$batch_size,
#         validation_split = 0.2, verbose = 0
#       )
#       gru_eval <- best_gru_model %>% evaluate(x_test, y_test, verbose = 0)
#       gru_mae <- c(gru_mae, gru_eval[[2]])
#       
#       best_gru_preds <- best_gru_model %>% predict(x_test)
#       gru_preds <- c(gru_preds, best_gru_preds)
#     }
#     
#     tibble(city = unique(.x$city), lstm_mae = mean(lstm_mae), gru_mae = mean(gru_mae), lstm_pred = lstm_preds, gru_pred = gru_preds)
#   })
# 
# print(results_by_city)

# Report results for each city
results_by_city <- climatedata %>%
  group_by(city) %>%
  group_modify(~ {
    lstm_mae <- numeric()
    gru_mae <- numeric()
    lstm_preds <- numeric()
    gru_preds <- numeric()
    
    k <- 5  # Number of cross-validation folds
    for (i in 1:k) {
      set.seed(123 * i)
      train_indices <- sample(seq_len(nrow(.x)), size = 0.8 * nrow(.x))
      train_data <- .x[train_indices, ]
      test_data <- .x[-train_indices, ]
      
      x_train_city <- as.matrix(train_data %>% select(tmin, tmax, sh, ws, rh, ep, rf))
      y_train_city <- as.matrix(train_data$sr)
      x_test_city <- as.matrix(test_data %>% select(tmin, tmax, sh, ws, rh, ep, rf))
      y_test_city <- as.matrix(test_data$sr)
      
      x_train_city <- array_reshape(x_train_city, c(nrow(x_train_city), ncol(x_train_city), 1))
      x_test_city <- array_reshape(x_test_city, c(nrow(x_test_city), ncol(x_test_city), 1))
      
      lstm_model_city <- build_lstm_model(
        units = best_lstm_result$units,
        optimizer = best_lstm_result$optimizer,
        activation = best_lstm_result$activation,
        batch_size = best_lstm_result$batch_size,
        epochs = best_lstm_result$epochs
      )$model
      gru_model_city <- build_gru_model(
        units = best_gru_result$units,
        optimizer = best_gru_result$optimizer,
        activation = best_gru_result$activation,
        batch_size = best_gru_result$batch_size,
        epochs = best_gru_result$epochs
      )$model
      
      lstm_model_city %>% fit(
        x = x_train_city, y = y_train_city,
        epochs = best_lstm_result$epochs, batch_size = best_lstm_result$batch_size,
        validation_split = 0.2, verbose = 0
      )
      lstm_eval_city <- lstm_model_city %>% evaluate(x_test_city, y_test_city, verbose = 0)
      lstm_preds_city <- lstm_model_city %>% predict(x_test_city)
      
      gru_model_city %>% fit(
        x = x_train_city, y = y_train_city,
        epochs = best_gru_result$epochs, batch_size = best_gru_result$batch_size,
        validation_split = 0.2, verbose = 0
      )
      gru_eval_city <- gru_model_city %>% evaluate(x_test_city, y_test_city, verbose = 0)
      gru_preds_city <- gru_model_city %>% predict(x_test_city)
      
      lstm_mae <- c(lstm_mae, lstm_eval_city[[2]])
      gru_mae <- c(gru_mae, gru_eval_city[[2]])
      lstm_preds <- c(lstm_preds, exp(lstm_preds_city))  # Reverse log transformation
      gru_preds <- c(gru_preds, exp(gru_preds_city))  # Reverse log transformation
    }
    
    tibble(
      city = unique(.x$city),
      lstm_mae = mean(lstm_mae),
      gru_mae = mean(gru_mae),
      lstm_preds = list(lstm_preds),
      gru_preds = list(gru_preds)
    )
  })

print("Results by city:")
print(results_by_city)

# Plot actual vs predicted values for each city
results_by_city %>%
  group_by(city) %>%
  do({
    plot_df <- data.frame(Actual = y_test_exp, Predicted = unlist(.$lstm_preds))
    ggplot(plot_df, aes(x = seq_along(Actual))) +
      geom_line(aes(y = Actual, color = "Actual")) +
      geom_line(aes(y = Predicted, color = "Predicted")) +
      labs(title = paste("LSTM: Actual vs Predicted for", unique(.$city)), x = "Index", y = "Values") +
      scale_color_manual(values = c("Actual" = "red", "Predicted" = "blue"))
  })

results_by_city %>%
  group_by(city) %>%
  do({
    plot_df <- data.frame(Actual = y_test_exp, Predicted = unlist(.$gru_preds))
    ggplot(plot_df, aes(x = seq_along(Actual))) +
      geom_line(aes(y = Actual, color = "Actual")) +
      geom_line(aes(y = Predicted, color = "Predicted")) +
      labs(title = paste("GRU: Actual vs Predicted for", unique(.$city)), x = "Index", y = "Values") +
      scale_color_manual(values = c("Actual" = "green", "Predicted" = "black"))
  })