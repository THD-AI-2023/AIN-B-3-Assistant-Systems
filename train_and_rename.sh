#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Function to handle errors
handle_error() {
    echo "An error occurred during training. Exiting."
    exit 1
}

# Trap errors and call the handle_error function
trap 'handle_error' ERR

# Validate the training data
echo "Starting Rasa data validation..."
rasa data validate

echo "Data validation passed successfully."

# Train the Rasa model
echo "Starting Rasa model training..."
python -m rasa train core --force

# Check if training was successful
if [ $? -ne 0 ]; then
    echo "Rasa training failed. Exiting."
    exit 1
fi

# Find the latest model file
LATEST_MODEL=$(ls -t /app/models/*.tar.gz | head -1)

# Check if a model was found
if [ -z "$LATEST_MODEL" ]; then
    echo "No model file found after training. Exiting."
    exit 1
fi

# Remove existing model.tar.gz if it exists
if [ -f /app/models/model.tar.gz ]; then
    rm /app/models/model.tar.gz
    echo "Existing model.tar.gz removed."
fi

# Rename the latest model to model.tar.gz
mv "$LATEST_MODEL" /app/models/model.tar.gz
echo "Model trained and renamed to model.tar.gz successfully."
