import streamlit as st
import torch
from PIL import Image
import json
import os
import pandas as pd
import numpy as np
from torch import nn
from torchvision import models, transforms
import io
from collections import defaultdict
import re
import joblib


# Define the feature extraction and model classes (same as you provided)
PATH_TO_MODEL = "C:/Users/nivl/Documents/Python/Projects/Complexity_prediction/streamlit/saved_models/best_model_val_loss_1.9360.pth"


# Load the saved scaler
scaler = joblib.load('C:/Users/nivl/Documents/Python/Projects/Complexity_prediction/streamlit/saved_models/scaler.pkl')

# Example of how to normalize the user-provided tabular features
def normalize_tabular_data(features):
    # Assuming `features` is a DataFrame
    features_normalized = scaler.transform(features)
    return pd.DataFrame(features_normalized, columns=features.columns)

# Define the feature extraction and model classes (same as you provided)
# Count items in the board appearing exactly 3 times
def count_triplets_board(data):
    board_items = data.get("board", [])
    return sum(1 for item in board_items if item["count"] == 3)

# Count items in the goals appearing exactly 3 times
def count_triplets_goals(data):
    goal_items = data.get("goals", [])
    return sum(1 for item in goal_items if item["count"] == 3)

# Remove trailing digits from item IDs
def get_base_name(item_id):
    return re.sub(r"\d+$", "", item_id)

# Analyze similar items in the level JSON
def count_similar_items(data):
    goals = {item["id"]: item["count"] for item in data.get("goals", [])}
    board = {item["id"]: item["count"] for item in data.get("board", [])}

    # Group items by their base name
    all_items = set(goals.keys()) | set(board.keys())
    grouped_items = defaultdict(list)
    for item_id in all_items:
        base_name = get_base_name(item_id)
        grouped_items[base_name].append(item_id)

    # Initialize counters
    total_similar_items_types = 0
    total_similar_items_number_items = 0
    similar_in_both_types = 0
    similar_in_both_number = 0

    for base_name, items in grouped_items.items():
        if len(items) > 1:
            # Calculate total count for this item type in goals and board
            in_goals_count = sum(goals.get(item, 0) for item in items if item in goals)
            in_board_count = sum(board.get(item, 0) for item in items if item in board)

            if in_goals_count > 0 or in_board_count > 0:
                total_similar_items_types += 1
                total_similar_items_number_items += in_goals_count + in_board_count

            if in_goals_count > 0 and in_board_count > 0:
                similar_in_both_types += 1
                similar_in_both_number += in_goals_count + in_board_count

    return total_similar_items_types, total_similar_items_number_items, similar_in_both_types, similar_in_both_number

# Feature engineering function
def calculate_features(input_json):
    """
    Calculate required features from the input JSON.
    """
    features = {}
    # Static data from JSON
    features["duration"] = input_json.get("duration", 0)
    features["has_ease"] = int(input_json.get("ease", 0) > 0)
    features["has_superEase_20"] = int(input_json.get("superEase", 0) == 20)
    features["assist"] = input_json.get("assist", 0)
    
    # Goals and board
    goals = input_json.get("goals", [])
    board = input_json.get("board", [])
    features["num_type_of_goals"] = len(goals)
    features["num_goal_items"] = sum(goal.get("count", 0) for goal in goals)
    features["total_items"] = sum(item.get("count", 0) for item in board) + sum(item.get("count", 0) for item in goals)
    features["num_board_items"] = sum(item.get("count", 0) for item in board)  
    
    # Triplet counts
    features["triplets_count_goal"] = count_triplets_goals(input_json)
    features["triplets_count_board"] = count_triplets_board(input_json)

    # Per second calculations    
    duration = features["duration"]
    # features["items_per_seconed"] = features["total_items"] / duration if duration > 0 else 0
    features["goal_items_per_seconed"] = features["num_goal_items"] / duration if duration > 0 else 0

    # Similar item analysis
    similar_items = count_similar_items(input_json)
    features["total_similar_items_types"] = similar_items[0]
    features["total_similar_items_number_items"] = similar_items[1]
    features["similar_in_both_types"] = similar_items[2]
    features["similar_in_both_number"] = similar_items[3]
    print("Features generated:", features.keys())
    print("Number of features:", len(features))
    return features


# The pretrained model class (MultimodalModel) will be the same as in your provided code
class MultimodalModel(nn.Module):
    def __init__(self, tabular_input_size, dropout_rate=0.3):
        super(MultimodalModel, self).__init__()
        self.dropout_rate = dropout_rate

        # Pretrained ResNet-50 backbone for image features
        self.image_model = models.resnet50(pretrained=True)
        self.image_model.fc = nn.Identity()  # Remove the classification head

        # Tabular model
        self.tabular_model = nn.Sequential(
            nn.Linear(tabular_input_size, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate)
        )

        # Combined model
        self.combined_model = nn.Sequential(
            nn.Linear(2048 + 128, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(128, 1)  # Regression output
        )

    def forward(self, image, tabular):
        # Image branch
        image_features = self.image_model(image)  # Extract 2048-dim features

        # Tabular branch
        tabular_features = self.tabular_model(tabular)  # Reduce to 128-dim features

        # Concatenate features from both branches
        combined_features = torch.cat((image_features, tabular_features), dim=1)

        # Process through the combined model
        output = self.combined_model(combined_features)
        return output
    
# Initialize the model
model = MultimodalModel(tabular_input_size=16)  # Assuming 13 features

# Ensure model is loaded on CPU, even if originally trained on GPU
model.load_state_dict(torch.load(PATH_TO_MODEL, map_location=torch.device('cpu')))  # Load the model to CPU
model.eval()

# Define the image transformation (resize and normalization)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# REQUIRED_FEATURE_ORDER for the input
REQUIRED_FEATURE_ORDER = [
    'duration', 
    'has_ease', 
    'has_superEase_20', 
    'assist', 
    'num_type_of_goals', 
    'num_goal_items', 
    'num_board_items', 
    'total_items', 
    'level_bin', 
    'goal_items_per_seconed', 
    'triplets_count_board', 
    'triplets_count_goal', 
    'total_similar_items_types', 
    'total_similar_items_number_items', 
    'similar_in_both_types', 
    'similar_in_both_number'
]

# Function to process the user inputs and make predictions
def make_prediction(input_json, uploaded_images, level_bin):
    # Extract features from JSON input
    features = calculate_features(input_json)
    
    # Add the selected level_bin to the features
    features['level_bin'] = level_bin
    
    # Prepare the tabular data (convert the features to a pandas DataFrame and then numpy array)
    tabular_data = pd.DataFrame([features])
    
    # Reorder the features according to REQUIRED_FEATURE_ORDER
    tabular_data = tabular_data[REQUIRED_FEATURE_ORDER]
    
    # Normalize the tabular data
    tabular_data_normalized = pd.DataFrame(scaler.transform(tabular_data), columns=tabular_data.columns)
    
    # Convert to tensor
    tabular_features = torch.tensor(tabular_data_normalized.values, dtype=torch.float32)

    # Process the uploaded images
    images = []
    for image_file in uploaded_images:
        image = Image.open(image_file).convert('RGB')
        image = transform(image)
        images.append(image)

    # If multiple images, average them
    if images:
        images = torch.stack(images).mean(dim=0).unsqueeze(0)  # Shape: (1, C, H, W)
    
    # Predict using the model
    with torch.no_grad():
        output = model(images, tabular_features)
    
    return output.item()

# Streamlit app
def main():
    st.title("Complexity Prediction")
    
    # Input: JSON for the level
    input_json = st.text_area("Enter JSON for the level:", height=300)
    
    # Input: Upload images
    uploaded_images = st.file_uploader("Upload Images (up to 3 images)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    
    # Input: Toggle for level_bin (0 to 19)
    level_bin = st.slider("Select Level Bin (0-19)", min_value=0, max_value=19, value=0)

    if st.button("Make Prediction"):
        if input_json and uploaded_images:
            try:
                # Parse JSON input
                input_json = json.loads(input_json)
                
                # Make prediction
                prediction = make_prediction(input_json, uploaded_images, level_bin)
                
                st.success(f"Prediction: {prediction:.2f}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.error("Please provide both the JSON and images.")

if __name__ == "__main__":
    main()