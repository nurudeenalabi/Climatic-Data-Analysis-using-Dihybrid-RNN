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
print(mean_summarize)
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
#full_data <- climatedata_ts[, 1:ncol(climatedata_ts)]

# Split the data into training and testing sets
set.seed(123) # For reproducibility
#train_indices <- sample(nrow(full_data), 0.8*nrow(full_data))
#train_data <- subset(full_data, rownames(full_data) %in% rownames(full_data)[train_indices])
#test_data <- subset(full_data, !(rownames(full_data) %in% rownames(full_data)[train_indices]))

#set.seed(123) # For reproducibility
#train_indices <- sample(nrow(full_data), 0.8 * nrow(full_data))
#train_data <- full_data[train_indices, ]
#test_data <- full_data[-train_indices, ]
train_indices <- sample(nrow(climatedata_ts), 0.8*nrow(climatedata_ts))
train_data <- subset(climatedata_ts, rownames(climatedata_ts) %in% rownames(climatedata_ts)[train_indices])
test_data <- subset(climatedata_ts, !(rownames(climatedata_ts) %in% rownames(climatedata_ts)[train_indices]))

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
#create_lstm_model <- function(input_shape) {
  #model <- keras_model_sequential()
  #model %>%
    #layer_lstm(units = 50, input_shape = input_shape) %>%
    #layer_dense(units = 1)
  
  #model %>% compile(optimizer = 'adam', loss = 'mse', 'rmse', 'mad')
  
  #return(model)
#}
# Define function to create LSTM model
create_lstm_model <- function(input_shape) {
  model <- keras_model_sequential()
  model %>% 
    layer_lstm(units = 50, input_shape = c(input_shape[2], input_shape[3])) %>% 
    layer_dense(units = 1)
  model %>% compile(optimizer = 'adam', loss = 'mse')
  return(model)
}

# Define function to create GRU model
#create_gru_model <- function(input_shape) {
 # model <- keras_model_sequential()
 # model %>%
 #   layer_gru(units = 50, input_shape = input_shape) %>%
 #   layer_dense(units = 1)
  
  #model %>% compile(optimizer = 'adam', loss = 'mse', 'mad', 'rmse', 'mape')
  
  #return(model)
#}
# Define function to create GRU model
create_gru_model <- function(input_shape) {
  model <- keras_model_sequential()
  model %>% 
    layer_gru(units = 50, input_shape = c(input_shape[2], input_shape[3])) %>% 
    layer_dense(units = 1)
  model %>% compile(optimizer = 'adam', loss = 'mse')
  return(model)
}

# Ensure correct Python path
correct_python_path <- "C:/Users/alabi/AppData/Local/Programs/Python/Python312/python.exe"

if (!file.exists(correct_python_path)) {
  stop("Python 3.12")
}

reticulate::use_python(correct_python_path)

library(keras)

# Function to create LSTM model
#create_lstm_model <- function(input_shape) {
  #inputs <- layer_input(shape = c(input_shape[2], input_shape[3]))
  
 # outputs <- inputs %>% 
 #   layer_lstm(units = 50) %>% 
  #  layer_dense(units = 1)
  
#  model <- keras_model(inputs = inputs, outputs = outputs)
  
 # return(model)
#}

#create_lstm_model <- function(input_shape) {
  #model <- keras_model_sequential()
  
 # model %>% 
  #  layer_input(batch_shape = c(input_shape[1], input_shape[2], input_shape[3])) %>% 
 #   layer_lstm(units = 50) %>% 
 #   layer_dense(units = 1)
  
#  model %>% compile(optimizer = 'adam', loss = 'mse')
  
#  return(model)
#}

# Example input shape (adjust as per your data)
input_shape <- list(32,10, 7)  # Example: (batch_size, timesteps, features)

# Create LSTM model
lstm_model <- create_lstm_model(input_shape)

# Compile the model
lstm_model %>% compile(optimizer = 'adam', loss = 'mse')

# Summary of the model
summary(lstm_model)


lstm_model <- create_lstm_model(input_shape)
gru_model <- create_gru_model(input_shape)

# Train LSTM model
lstm_history <- lstm_model %>% fit(X_train, y_train, epochs = 1, batch_size = 32, validation_split = 0.2, verbose = 2)

# Train GRU model
gru_history <- gru_model %>% fit(X_train, y_train, epochs = 1, batch_size = 32, validation_split = 0.2, verbose = 2)

# Plot training history
plot(lstm_history)
plot(gru_history)

#City Specific models
# Create a list to store city-specific models
city_models <- list()

# Loop through each city
for (city in unique(climatedata_wide$city)) {
  # Subset data for the current city
  city_data <- climatedata_wide[climatedata_wide$city == city, ]
  
  # Prepare input data for LSTM and GRU models
  X_train <- array(city_data[, -1], dim = c(nrow(city_data), ncol(city_data) - 1, 1))
  y_train <- city_data[, 1]
  
  # Create LSTM model for the current city
  lstm_model <- create_lstm_model(c(ncol(city_data) - 1, 1))
  lstm_history <- lstm_model %>% fit(X_train, y_train, epochs = 1, batch_size = 32, validation_split = 0.2, verbose = 2)
  
  # Create GRU model for the current city
  gru_model <- create_gru_model(c(ncol(city_data) - 1, 1))
  gru_history <- gru_model %>% fit(X_train, y_train, epochs = 1, batch_size = 32, validation_split = 0.2, verbose = 2)
  
  # Store the city-specific models in the list
  city_models[[city]] <- list(lstm = lstm_model, gru = gru_model)
}

# Create a list to store city-specific forecasts
city_forecasts <- list()

# Loop through each city
for (city in names(city_models)) {
  # Get the city-specific models
  lstm_model <- city_models[[city]]$lstm
  gru_model <- city_models[[city]]$gru
  
  # Prepare input data for forecasting
  X_forecast <- array(city_data[, -1], dim = c(nrow(city_data), ncol(city_data) - 1, 1))
  
  # Forecast using LSTM and GRU models
  lstm_forecast <- predict(lstm_model, X_forecast)
  gru_forecast <- predict(gru_model, X_forecast)
  
  # Store the city-specific forecasts in the list
  city_forecasts[[city]] <- list(lstm = lstm_forecast, gru = gru_forecast)
}

