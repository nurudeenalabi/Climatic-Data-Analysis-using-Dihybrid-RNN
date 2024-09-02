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
# Configure R to use the r-reticulate conda environment
conda_path <- "C:/Users/alabi/Anaconda3/Scripts/conda.exe"
Sys.setenv(RETICULATE_CONDA = conda_path)
use_condaenv("r-reticulate", required = TRUE)
tf$constant("Hello, TensorFlow!")
tf$compat$v1$ConfigProto(gpu_options = tf$compat$v1$GPUOptions(allow_growth = TRUE))
# Convert 'Year' to datetime and set it as index
climatedata <- climatedata %>%
  mutate(Year = ymd(paste0(Year, "-01-01"))) %>%
  arrange(Year)

# Remove non-numeric values and convert to numeric
climatedata <- climatedata %>%
  mutate(across(-Year, ~ as.numeric(gsub("[^0-9.-]", "", .))))

# Fill NA values introduced by coercion using linear interpolation
climatedata <- climatedata %>%
  mutate(across(-Year, ~ zoo::na.approx(., na.rm = FALSE))) %>%
  fill(everything(), .direction = "down")

# Apply natural log transformation to the data (excluding 'Year' and 'sr')
climatedata_log <- climatedata %>%
  mutate(across(c(tmin, tmax, sh, rf, ep, rh, ws), ~ log(. + 1)))  # Adding 1 to avoid log(0)

# Normalize the data
features <- c("tmin", "tmax", "sh", "rf", "ep", "rh", "ws")
data_normalized <- climatedata_log %>%
  select(all_of(features)) %>%
  scale() %>%
  as_tibble()
data_normalized <- bind_cols(climatedata %>% select(Year, sr), data_normalized)

# Split the data into training and test sets
set.seed(123)
train_indices <- sample(seq_len(nrow(data_normalized)), size = 0.8 * nrow(data_normalized))
train_data <- data_normalized[train_indices, ]
test_data <- data_normalized[-train_indices, ]

# Convert data to matrices
train_x <- as.matrix(train_data %>% select(-Year, -sr))
train_y <- as.matrix(train_data$sr)
test_x <- as.matrix(test_data %>% select(-Year, -sr))
test_y <- as.matrix(test_data$sr)

# Define the model structure
build_model <- function(optimizer) {
  model <- keras_model_sequential() %>%
    layer_dense(units = 64, activation = "relu", input_shape = ncol(train_x)) %>%
    layer_dense(units = 32, activation = "relu") %>%
    layer_dense(units = 1)
  
  model %>% compile(
    loss = "mean_squared_error",
    optimizer = optimizer,
    metrics = c("mean_absolute_error")
  )
  
  return(model)
}

# Train and evaluate the model with different optimizers
optimizers <- list("adam", "rmsprop", "sgd")
results <- list()

for (opt in optimizers) {
  model <- build_model(opt)
  
  history <- model %>% fit(
    train_x, train_y,
    epochs = 100,
    validation_split = 0.2,
    verbose = 1
  )
  
  scores <- model %>% evaluate(test_x, test_y, verbose = 0)
  results[[opt]] <- scores
}

# Print the results
results

# Load necessary libraries
library(keras)
library(ggplot2)

# Assuming 'test_x' is your test dataset input
# Assuming 'test_y' is your actual test labels

# Make predictions
predictions <- model %>% predict(test_x)

# Convert predictions to a vector for comparison (if necessary)
predictions <- as.vector(predictions)
print(predictions)
# Create a data frame to compare actual vs predicted values
results <- data.frame(
  Actual = as.vector(test_y),
  Predicted = predictions
)

# Calculate the mean absolute error (MAE) for the test set
mae <- mean(abs(results$Actual - results$Predicted))
cat('Test Mean Absolute Error (MAE):', mae, '\n')

# Plot the results
ggplot(results, aes(x = Actual, y = Predicted)) +
  geom_point(color = 'blue') +
  geom_abline(intercept = 0, slope = 1, color = 'red', linetype = 'dashed') +
  labs(title = 'Actual vs Predicted', x = 'Actual Values', y = 'Predicted Values') +
  theme_minimal()

# Plot actual vs predicted values over time
results$Index <- 1:nrow(results)  # Assuming sequential data

ggplot(results, aes(x = Index)) +
  geom_line(aes(y = Actual, color = "Actual")) +
  geom_line(aes(y = Predicted, color = "Predicted")) +
  labs(title = 'Actual vs Predicted over Time', x = 'Index', y = 'Values') +
  scale_color_manual(name = "Legend", values = c("Actual" = "blue", "Predicted" = "green")) +
  theme_minimal()

# Calculate residuals
results$residuals <- results$Actual - results$Predicted

# Plot residuals
ggplot(results, aes(x = Index, y = residuals)) +
  geom_line(color = 'red') +
  labs(title = 'Residuals over Time', x = 'Index', y = 'Residuals') +
  theme_minimal()

# Plot histogram of residuals
ggplot(results, aes(x = residuals)) +
  geom_histogram(bins = 30, fill = 'blue', alpha = 0.7) +
  labs(title = 'Histogram of Residuals', x = 'Residuals', y = 'Frequency') +
  theme_minimal()

# Load necessary library
library(forecast)

# Plot ACF of residuals
acf(results$residuals, main = 'Autocorrelation of Residuals')





#Models on each city
library(keras)
library(tensorflow)
library(tidyverse)

# Initialize lists to store models and results
city_models <- list()
city_results <- list()

# Loop through each city's dataset
for (i in seq_along(climatedata)) {
  city_data <- climatedata[[i]]
  city_name <- unique(city_data$city)
  
  # Prepare data (this is a simplified example, adjust as needed)
  x_train <- city_data %>% select(-solar_radiation) %>% as.matrix()
  y_train <- city_data$solar_radiation
  
  # Define the model
  model <- keras_model_sequential() %>%
    layer_lstm(units = 50, input_shape = c(ncol(x_train), 1), return_sequences = TRUE) %>%
    layer_dropout(rate = 0.2) %>%
    layer_lstm(units = 50, return_sequences = FALSE) %>%
    layer_dropout(rate = 0.2) %>%
    layer_dense(units = 1)
  
  model %>% compile(
    loss = 'mean_squared_error',
    optimizer = optimizer_adam(),
    metrics = c('mean_absolute_error')
  )
  
  # Train the model
  history <- model %>% fit(
    x_train, y_train,
    epochs = 100,
    batch_size = 32,
    validation_split = 0.2
  )
  
  # Store the model and training history
  city_models[[city_name]] <- model
  city_results[[city_name]] <- history
}

# Save the models for later use
save(city_models, file = "city_models.RData")
save(city_results, file = "city_results.RData")
# Example of plotting training history for each city
par(mfrow = c(2, 1))
for (city_name in names(city_results)) {
  history <- city_results[[city_name]]
  
  # Plot training and validation loss
  plot(history$metrics$loss, type = 'l', col = 'blue', ylim = range(history$metrics$loss, history$metrics$val_loss), 
       ylab = 'Loss', xlab = 'Epoch', main = paste(city_name, 'Loss'))
  lines(history$metrics$val_loss, col = 'green')
  
  # Plot training and validation mean absolute error
  plot(history$metrics$mean_absolute_error, type = 'l', col = 'blue', ylim = range(history$metrics$mean_absolute_error, history$metrics$val_mean_absolute_error), 
       ylab = 'Mean Absolute Error', xlab = 'Epoch', main = paste(city_name, 'MAE'))
  lines(history$metrics$val_mean_absolute_error, col = 'green')
}

