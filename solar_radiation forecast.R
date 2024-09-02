# Load necessary libraries
library(keras)
library(tidyverse)

# Set the working directory
setwd("C:/Users/alabi/OneDrive/Desktop")

# Load climate data
climatedata <- read.csv("climate_data.csv", header = TRUE, na.strings = "?")

# Check the structure of climatedata dataframe
str(climatedata)

# Remove columns that are entirely NA
climatedata <- climatedata %>% select(where(~ !all(is.na(.))))

# Print the structure again to verify
str(climatedata)

# Define a function to summarize duplicate values by taking the mean
mean_summarize <- function(x) mean(as.numeric(x), na.rm = TRUE)

# Extract month and year from the period column
climatedata <- climatedata %>%
  mutate(month = substr(period, 1, 3))

# Group by city and month, then summarize to avoid duplicates
climatedata_grouped <- climatedata %>%
  group_by(city, month) %>%
  summarize(
    tmin = mean_summarize(tmin),
    tmax = mean_summarize(tmax),
    rh = mean_summarize(rh),
    sh = mean_summarize(sh),
    ep = mean_summarize(ep),
    ws = mean_summarize(ws),
    rf = mean_summarize(rf),
    .groups = 'drop'
  )

# Pivot the data wider
climatedata_wide <- pivot_wider(
  data = climatedata_grouped, 
  names_from = month, 
  values_from = c(tmin, tmax, rh, sh, ep, ws, rf)
)

# Convert the summarized pivoted data to matrix format, excluding the city column
climatedata_ts <- as.matrix(climatedata_wide[, -1])

# Check dimensions before splitting
print(dim(climatedata_ts))

# Split data into training and testing sets
#train_size <- floor(0.8 * nrow(climatedata_ts))
#train_data <- climatedata_ts[1:train_size, ]
#test_data <- climatedata_ts[(train_size + 1):nrow(climatedata_ts), 1:ncol(climatedata_ts)]

# Create a new dataset with all rows and columns
full_data <- climatedata_ts[, 1:ncol(climatedata_ts)]

# Split the data into training and testing sets
set.seed(123) # For reproducibility
train_indices <- sample(nrow(full_data), 0.8*nrow(full_data))
train_data <- subset(full_data, rownames(full_data) %in% rownames(full_data)[train_indices])
test_data <- subset(full_data, !(rownames(full_data) %in% rownames(full_data)[train_indices]))


# Check dimensions after splitting
print(dim(train_data))
print(dim(test_data))

# Ensure train_data and test_data have the correct number of columns
stopifnot(ncol(train_data) == ncol(test_data))

# Check column names in train and test data
print(colnames(train_data))
print(colnames(test_data))

# Calculate max and min values for normalization from training data only
max_vals <- apply(train_data, 2, max, na.rm = TRUE)
min_vals <- apply(train_data, 2, min, na.rm = TRUE)

# Scale the data
scaled_train_data <- scale(train_data, center = min_vals, scale = max_vals - min_vals)

# Attempt to scale test data, with additional checks
scaled_test_data <- tryCatch({
  scale(test_data, center = min_vals, scale = max_vals - min_vals)
}, error = function(e) {
  print("Error in scaling test data:")
  print(e)
  return(NULL)
})

# Check if scaling was successful
if (is.null(scaled_test_data)) {
  stop("Scaling of test data failed. Check the dimensions and columns of test data.")
}

# Prepare training and testing datasets
X_train <- scaled_train_data[, -1]
y_train <- scaled_train_data[, 1]
X_test <- scaled_test_data[, -1]
y_test <- scaled_test_data[, 1]

# Reshape input data for LSTM and GRU
input_shape <- c(ncol(X_train), 1)
X_train <- array(X_train, dim = c(nrow(X_train), input_shape))
X_test <- array(X_test, dim = c(nrow(X_test), input_shape))

# Define function to create LSTM model
create_lstm_model <- function(input_shape) {
  model <- keras_model_sequential()
  model %>%
    layer_lstm(units = 50, input_shape = input_shape) %>%
    layer_dense(units = 1)
  
  model %>% compile(optimizer = 'adam', loss = 'mse')
  
  return(model)
}

# Define function to create GRU model
create_gru_model <- function(input_shape) {
  model <- keras_model_sequential()
  model %>%
    layer_gru(units = 50, input_shape = input_shape) %>%
    layer_dense(units = 1)
  
  model %>% compile(optimizer = 'adam', loss = 'mse')
  
  return(model)
}

# Create LSTM and GRU models
lstm_model <- create_lstm_model(input_shape)
gru_model <- create_gru_model(input_shape)

# Train LSTM model
lstm_history <- lstm_model %>% fit(X_train, y_train, epochs = 100, batch_size = 32, validation_split = 0.2, verbose = 2)

# Train GRU model
gru_history <- gru_model %>% fit(X_train, y_train, epochs = 100, batch_size = 32, validation_split = 0.2, verbose = 2)

# Plot training history
plot(lstm_history)
plot(gru_history)
