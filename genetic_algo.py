import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
import sklearn.metrics
import random

# Read the dataset
data = pd.read_csv('Hotel Reservations.csv')

def preprocess_data(data):
	# Remove columns that are not needed for the analysis
	data = data.drop('Booking_ID', axis = 1)
	data = data.drop('market_segment_type', axis = 1) 
	data = data.drop('arrival_year', axis = 1)
	data = data.drop('lead_time', axis = 1)
	
	# Balance dataset by undersampling majority class
	positive_samples = data[data['booking_status'] == 1]
	negative_samples = data[data['booking_status'] == 0]
	
	# Get the size of the minority class
	min_size = min(len(positive_samples), len(negative_samples))
	
	# Randomly sample from majority class to match minority class size
	if len(positive_samples) > len(negative_samples):
		positive_samples = positive_samples.sample(n=min_size, random_state=42)
		balanced_data = pd.concat([positive_samples, negative_samples])
	else:
		negative_samples = negative_samples.sample(n=min_size, random_state=42)
		balanced_data = pd.concat([positive_samples, negative_samples])
	
	return balanced_data

# Preprocess the data
data  = preprocess_data(data)

# Split data into training (70%) and test (30%) sets
train_data = data.sample(frac=0.7, random_state=4)  # random_state ensures reproducibility
test_data = data.drop(train_data.index)	

# Separate features (x) and target variable (y) for both training and test sets
x_train = train_data.drop('booking_status', axis = 1)  # Features for training
x_test = test_data.drop('booking_status', axis = 1)    # Features for testing
y_train = train_data['booking_status']                 # Target variable for training
y_test = test_data['booking_status']                   # Target variable for testing