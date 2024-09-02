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
print(climatedata)

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

# Identify non-numeric values
non_numeric_values <- climatedata %>%
  select(-Year) %>%
  map(~ sum(!grepl("^-?\\d*(\\.\\d+)?$", .))) %>%
  keep(~ . > 0)
print(non_numeric_values)

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

# Check for NA or infinite values in the normalized data
summary(data_normalized)

# Print first few rows of the normalized data to verify
head(data_normalized)


# Create sequences for LSTM
create_sequences <- function(data, seq_length, features) {
  X <- array(dim = c(nrow(data) - seq_length, seq_length, length(features)))
  y <- array(dim = c(nrow(data) - seq_length))
  for (i in seq_len(nrow(data) - seq_length)) {
    X[i,,] <- as.matrix(data[i:(i + seq_length - 1), features])
    y[i] <- data$sr[i + seq_length]  # Adjust this to the correct target variable
  }
  list(X, y)
}

seq_length <- 5
sequences <- create_sequences(data_normalized, seq_length, features)
X <- sequences[[1]]
y <- sequences[[2]]

# Split the data into training and test sets
split <- floor(0.8 * nrow(X))
X_train <- X[1:split,,]
X_test <- X[(split + 1):nrow(X),,]
y_train <- y[1:split]
y_test <- y[(split + 1):length(y)]

# Define the input shape
input_shape <- c(seq_length, length(features))

# Define a function to build and train the model with a specified optimizer
train_model <- function(optimizer, X_train, y_train, X_test, y_test, input_shape) {
  input_layer <- layer_input(shape = input_shape)
  lstm_layer <- layer_lstm(units = 10)(input_layer)
  dropout_layer <- layer_dropout(rate = 0.2)(lstm_layer)
  output_layer <- layer_dense(units = 1)(dropout_layer)
  
  model <- keras_model(inputs = input_layer, outputs = output_layer)
  
  model %>% compile(
    loss = "mse",
    optimizer = optimizer,
    metrics = "mae"
  )
  
  history <- model %>% fit(
    X_train, y_train,
    epochs = 10,
    batch_size = 16,
    validation_split = 0.2
  )
  
  evaluation <- model %>% evaluate(X_test, y_test)
  predictions <- model %>% predict(X_test)
  
  list(model = model, history = history, evaluation = evaluation, predictions = predictions)
}

# Adjust learning rates and optimizers
optimizers <- list(
  sgd = optimizer_sgd(lr = 0.001, momentum = 0.9),
  rmsprop = optimizer_rmsprop(lr = 0.0001),
  adagrad = optimizer_adagrad(lr = 0.001),
  adadelta = optimizer_adadelta(lr = 0.1),
  adam = optimizer_adam(lr = 0.0001),
  adamax = optimizer_adamax(lr = 0.0002),
  nadam = optimizer_nadam(lr = 0.0002)
)

# Train models with adjusted configuration
results <- lapply(optimizers, function(opt) {
  train_model(opt, X_train, y_train, X_test, y_test, input_shape)
})

# Print the results
for (opt_name in names(results)) {
  cat("\nOptimizer:", opt_name, "\n")
  cat("Test Loss:", results[[opt_name]]$evaluation[[1]], "\n")
  cat("Test MAE:", results[[opt_name]]$evaluation[[2]], "\n")
}
