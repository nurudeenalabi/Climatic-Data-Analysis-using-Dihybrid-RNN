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
# Update the model building functions
build_lstm_model <- function(units = 50, learning_rate = 0.001, optimizer = 'adam', activation = 'relu', recurrent_activation = 'gelu', batch_size = 32, epochs = 100) {
  input_layer <- layer_input(shape = c(ncol(x_train), 1))
  lstm_layer <- layer_lstm(units = units, activation = activation, return_sequences = TRUE)(input_layer)
  lstm_layer <- layer_lstm(units = units, activation = activation)(lstm_layer)
  output_layer <- layer_dense(units = 1, activation = 'linear')(lstm_layer)
  
  model <- keras_model(inputs = input_layer, outputs = output_layer)
  model %>% compile(
    loss = 'mean_squared_error',
    optimizer = optimizer,
    metrics = list('mean_absolute_error', custom_metric("rmse", rmse), custom_metric("rsquare", rsquare))
  )
  return(list(model = model, batch_size = batch_size, epochs = epochs))
}
build_gru_model <- function(units = 50, learning_rate = 0.001, optimizer = 'adam', activation = 'tanh', recurrent_activation = 'hard_tanh', batch_size = 32, epochs = 100) {
  input_layer <- layer_input(shape = c(ncol(x_train), 1))
  gru_layer <- layer_gru(units = units, activation = activation, return_sequences = TRUE)(input_layer)
  gru_layer <- layer_gru(units = units, activation = activation)(gru_layer)
  output_layer <- layer_dense(units = 1, activation = 'linear')(gru_layer)
  
  model <- keras_model(inputs = input_layer, outputs = output_layer)
  model %>% compile(
    loss = 'mean_squared_error',
    optimizer = optimizer,
    metrics = list('mean_absolute_error', custom_metric("rmse", rmse), custom_metric("rsquare", rsquare))
  )
  return(list(model = model, batch_size = batch_size, epochs = epochs))
}
# # Define a smaller hyperparameter grid for debugging
# hyper_grid <- expand.grid(
#   units = c(50),
#   learning_rate = c(0.0001,0.001, 0.01, 0.1),
#   batch_size = c(32,64),  # Smaller batch size for faster execution
#   epochs = c(100)       # Fewer epochs for debugging
# )
# Define an expanded hyperparameter grid
# hyper_grid <- expand.grid(
#   units = c(50, 100, 150),
#   learning_rate = c(0.0001, 0.001, 0.01, 0.1),
#   batch_size = c(32, 64, 128),
#   epochs = c(50, 100, 150)
# )
hyper_grid <- expand.grid(
  units = c( 150),
  learning_rate = c(0.00001),
  batch_size = c(128),
  epochs = c(150), verbose = 2
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
      c(loss = evaluation[[1]], mae = evaluation[[2]], rmse = evaluation[[3]], rsquare = evaluation[[4]])
    })
    cv_results <- t(cv_results)
    colnames(cv_results) <- c("loss", "mae", "rmse", "rsquare")
    results[[params]] <- c(model_params, colMeans(cv_results, na.rm = TRUE))
  }
  
  # Convert results to DataFrame
  results_df <- do.call(rbind, results)
  results_df <- as.data.frame(results_df)
  # Set column names
  colnames(results_df) <- c(names(hyper_grid), "loss", "mae", "rmse", "rsquare")
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
gru_results
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

best_gru_model_info$model%>%summary()
summary(best_gru_model_info$model)
# Train the best GRU model on the full training data
best_gru_model_info$model %>% fit(
  x = x_train, y = y_train,
  epochs = epochs, batch_size = batch_size,
  validation_split = 0.2, verbose = 0
)

# Evaluate on test data
gru_eval <- best_gru_model_info$model %>% evaluate(x_test, y_test, verbose = 0)
best_gru_preds <- best_gru_model_info$model %>% predict(x_test)

# Reverse the log transformation
y_test_exp <- exp(y_test)
best_lstm_preds_exp <- exp(best_lstm_preds)
best_gru_preds_exp <- exp(best_gru_preds)
# library(shapper)
# library(iml)
# library(reticulate)
# shap <- import("shap")
# explainer <- shap$TreeExplainer(best_lstm_result)
# shap_values <- explainer$shap_values(X_train)
# shap$summary_plot(shap_values, X_train)
# library(keras)
# library(tidyverse)
# library(reticulate)
# shap <- import("shap")
# 
# # Assuming your model is already trained and named `model`
# # Assuming your data is named `X_train`
# 
# # Convert data to numpy array if needed
# np <- import("numpy")
# X_train_np <- np$array(X_train)
# 
# # Generate SHAP values
# explainer <- shap$DeepExplainer(model, X_train_np)
# shap_values <- explainer$shap_values(X_train_np)
# 
# # Plot SHAP summary
# shap$summary_plot(shap_values, X_train_np)

# Plot actual vs predicted values for the best LSTM model
plot_df <- data.frame(Actual = y_test_exp, Predicted = best_lstm_preds_exp)
lstm_plot = ggplot(plot_df, aes(x = seq_along(Actual))) +
  geom_line(aes(y = Actual, color = "Actual")) +
  geom_line(aes(y = Predicted, color = "Predicted")) +
  labs(title = "LSTM: Actual vs Predicted", x = "Index", y = "Values") +
  scale_color_manual(values = c("Actual" = "red", "Predicted" = "blue"))

# Plot actual vs predicted values for the best GRU model
plot_df <- data.frame(Actual = y_test_exp, Predicted = best_gru_preds_exp)
gru_plot = ggplot(plot_df, aes(x = seq_along(Actual))) +
  geom_line(aes(y = Actual, color = "Actual")) +
  geom_line(aes(y = Predicted, color = "Predicted")) +
  labs(title = "GRU: Actual vs Predicted", x = "Index", y = "Values") +
  scale_color_manual(values = c("Actual" = "green", "Predicted" = "black"))
plot_grid(lstm_plot,gru_plot, nrow =1, ncol = 2)

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
      
      # Calculate RMSE
      lstm_rmse <- c(lstm_rmse, sqrt(mean((y_test_city - lstm_preds_city)^2)))
      gru_rmse <- c(gru_rmse, sqrt(mean((y_test_city - gru_preds_city)^2)))
      
      # Calculate R²
      lstm_rsquare <- c(lstm_rsquare, 1 - (sum((y_test_city - lstm_preds_city)^2) / sum((y_test_city - mean(y_test_city))^2)))
      gru_rsquare <- c(gru_rsquare, 1 - (sum((y_test_city - gru_preds_city)^2) / sum((y_test_city - mean(y_test_city))^2)))
      
      lstm_mae <- c(lstm_mae, lstm_eval_city[[2]])
      gru_mae <- c(gru_mae, gru_eval_city[[2]])
      lstm_preds <- c(lstm_preds, exp(lstm_preds_city))  # Reverse log transformation
      gru_preds <- c(gru_preds, exp(gru_preds_city))  # Reverse log transformation
    }
    
    tibble(
      city = unique(.x$city),
      lstm_mae = mean(lstm_mae),
      gru_mae = mean(gru_mae),
      lstm_rmse = mean(lstm_rmse),
      gru_rmse = mean(gru_rmse),
      lstm_rsquare = mean(lstm_rsquare),
      gru_rsquare = mean(gru_rsquare),
      lstm_preds = list(lstm_preds),
      gru_preds = list(gru_preds)
    )
  })

results_by_city

library(dplyr)
library(ggplot2)

# Function to reverse log transformation
reverse_log_transform <- function(x) {
  return(exp(x))
}

# Assuming `climatedata` and models are defined, and `results_by_city` contains necessary results
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
      
      # Calculate RMSE
      lstm_rmse <- c(lstm_rmse, sqrt(mean((y_test_city - lstm_preds_city)^2)))
      gru_rmse <- c(gru_rmse, sqrt(mean((y_test_city - gru_preds_city)^2)))
      
      # Calculate R²
      lstm_rsquare <- c(lstm_rsquare, 1 - (sum((y_test_city - lstm_preds_city)^2) / sum((y_test_city - mean(y_test_city))^2)))
      gru_rsquare <- c(gru_rsquare, 1 - (sum((y_test_city - gru_preds_city)^2) / sum((y_test_city - mean(y_test_city))^2)))
      
      lstm_mae <- c(lstm_mae, lstm_eval_city[[2]])
      gru_mae <- c(gru_mae, gru_eval_city[[2]])
      lstm_preds <- c(lstm_preds, exp(lstm_preds_city))  # Reverse log transformation
      gru_preds <- c(gru_preds, exp(gru_preds_city))  # Reverse log transformation
    }
    
    tibble(
      city = unique(.x$city),
      lstm_mae = mean(lstm_mae),
      gru_mae = mean(gru_mae),
      lstm_rmse = mean(lstm_rmse),
      gru_rmse = mean(gru_rmse),
      lstm_rsquare = mean(lstm_rsquare),
      gru_rsquare = mean(gru_rsquare),
      lstm_preds = list(lstm_preds),
      gru_preds = list(gru_preds)
    )
  })

# Define reverse_log_transform function if not already defined
reverse_log_transform <- function(x) {
  exp(x)
}

# Define reverse_log_transform function if not already defined
reverse_log_transform <- function(x) {
  exp(x)
}

# Plotting the results
plots <- lapply(unique(results_by_city$city), function(city) {
  city_data <- results_by_city %>% filter(city == !!city)
  
  actual_values <- climatedata %>% filter(city == !!city) %>% pull(sr)
  predicted_values <- reverse_log_transform(as.numeric(unlist(city_data$lstm_preds)))
  
  # Ensure both vectors have the same length
  min_length <- min(length(actual_values), length(predicted_values))
  actual_values <- actual_values[1:min_length]
  predicted_values <- predicted_values[1:min_length]
  
  plot_df <- data.frame(
    Actual = actual_values,
    Predicted = predicted_values
  )
  
  ggplot(plot_df, aes(x = seq_along(Actual))) +
    geom_line(aes(y = Actual, color = "Actual")) +
    geom_line(aes(y = Predicted, color = "Predicted")) +
    labs(title = paste("LSTM: Actual vs Predicted for", city), x = "Index", y = "Values") +
    scale_color_manual(values = c("Actual" = "red", "Predicted" = "blue"))
})

# If you want to print or display the plots
print(plots)
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

# Display the combined grid of plots
print(combined_plots)