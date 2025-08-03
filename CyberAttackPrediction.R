# Load required libraries
library(data.table)   # For efficient data handling
library(caret)        # For machine learning tasks like data partitioning and preprocessing
library(xgboost)      # For training the XGBoost model
library(pROC)         # For calculating ROC and AUC
library(ggplot2)      # For plotting (not directly used here, but good to have for visualization)

# Load feature names from the feature info CSV file
feature_info <- fread("NUSW-NB15_features.csv", header = TRUE, sep = ",")

# Extract the names and replace "Label" with "label" to standardize
feature_names <- c(feature_info$Name)
feature_names[feature_names == "Label"] <- "label"
print(feature_names)

# Load the first 40,000 rows of the UNSW-NB15 dataset without headers
data <- fread("UNSW-NB15_1.csv", nrows = 40000, header = FALSE)

# Assign proper column names using the feature_names list
setnames(data, feature_names)
colnames(data)

# Remove rows with missing values
data <- na.omit(data)

# If "attack_cat" column exists, remove it to prevent data leakage
if ("attack_cat" %in% colnames(data)) {
  data <- data[, !c("attack_cat"), with = FALSE]
}

# Convert categorical columns to factors
cat_cols <- c("proto", "service", "state")
for (col in cat_cols) {
  if (col %in% names(data)) {
    data[[col]] <- as.factor(data[[col]])
  }
}

# Ensure column names are valid and unique
names(data) <- make.names(names(data), unique = TRUE)

# Print the label column
print(data$label)

# Ensure the label column is a factor (for classification)
data$label <- as.factor(data$label)

# Remove columns with only one unique value (no useful information)
constant_cols <- names(Filter(function(x) length(unique(x)) == 1, data))
data <- data[, !..constant_cols]

# Check again if label is a factor (just in case)
if (!is.factor(data$label)) {
  data$label <- as.factor(data$label)
}

# One-Hot Encoding: Convert all categorical variables to dummy variables (binary columns)
dummies <- dummyVars(~ ., data = data[, !"label", with = FALSE])
X <- predict(dummies, newdata = data[, !"label", with = FALSE])
X <- as.data.frame(X)

# Extract the target variable
y <- data$label

# Split the dataset: 80% training and 20% testing
set.seed(42)
trainIndex <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[trainIndex, ]
X_test  <- X[-trainIndex, ]
y_train <- y[trainIndex]
y_test  <- y[-trainIndex]

# Convert training and test data to xgboost DMatrix format
# Also convert factor levels to numeric (0/1 for binary classification)
dtrain <- xgb.DMatrix(data = as.matrix(X_train), label = as.numeric(y_train) - 1)
dtest  <- xgb.DMatrix(data = as.matrix(X_test), label = as.numeric(y_test) - 1)

# Set XGBoost parameters for binary classification
params <- list(
  objective = "binary:logistic",  # Binary classification problem
  eval_metric = "logloss",        # Evaluation metric
  booster = "gbtree"              # Use tree-based model
)

# Train the XGBoost model with 100 boosting rounds
xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 100,
  watchlist = list(train = dtrain, eval = dtest),
  verbose = 0                    # Suppress training output
)

# Predict probabilities for the test set
pred_probs <- predict(xgb_model, dtest)

# Convert probabilities to class labels (threshold = 0.5)
pred_labels <- ifelse(pred_probs > 0.5, 1, 0)

# Generate confusion matrix to evaluate classification performance
conf_mat <- confusionMatrix(as.factor(pred_labels), as.factor(as.numeric(y_test) - 1))
print(conf_mat)

# Calculate and plot feature importance
importance <- xgb.importance(model = xgb_model)
xgb.plot.importance(importance_matrix = importance, top_n = 20, main = "Feature Importance")

# Compute ROC curve and AUC score
roc_obj <- roc(as.numeric(y_test) - 1, pred_probs)
auc_score <- auc(roc_obj)
cat("AUC Score:", auc_score, "\n")

# Plot the ROC curve
plot(roc_obj, main = sprintf("ROC Curve (AUC = %.3f)", auc_score), col = "blue", lwd = 2)
abline(a = 0, b = 1, lty = 2, col = "gray")  # Add diagonal reference line
