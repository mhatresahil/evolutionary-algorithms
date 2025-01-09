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

# Fitness Function for Genetic Algorithm
def fitness_func(model, weights, x, y):
    # Reshape weights directly into layer shapes
    layer1_weights = [weights[:14*28].reshape((14, 28)), weights[14*28:420]]
    layer2_weights = [weights[420:-1].reshape((28,1)), weights[-1].reshape((1,))]
    
    # Set weights for both layers at once
    model.layers[0].set_weights(layer1_weights)
    model.layers[1].set_weights(layer2_weights)
    
    # Evaluate with less verbosity
    score = model.evaluate(x, y, verbose=0)
    return score[1]

#Genetic Algorithm
def genetic_algo(model, fitness, population_size, x, y, max_limit):
    # Initialize population with masked weights
    weights = model.get_weights()
    weights_vector = np.concatenate([w.flatten() for w in weights])
    population = [weights_vector * np.random.choice([0, 1], size=weights_vector.shape, p=[0.9, 0.1]) 
                 for _ in range(population_size)]
    
    for _ in range(max_limit):
        new_population = []
        #fitness-proportionate selection
        tfitness = [fitness(model, w, x, y) for w in population]
        total_fit = sum(tfitness)
        probabilities = [score / total_fit for score in tfitness]
        for j in range(population_size//2):
            p1_index = np.random.choice(len(tfitness), replace = False, p = probabilities)
            p2_index = np.random.choice(len(tfitness), replace = False, p = probabilities)
            p1 = population[p1_index]
            p2 = population[p2_index]
            
            #Generate Offsprings
            child = p1.copy()
            child1 = p2.copy()
            
            #Uniform Crossover with 90% probability
            if random.choices([True, False], weights=[0.9, 0.1])[0]:
                par_shape = p1.shape
                ux = np.random.randint(low=0, high=2, size=par_shape).astype(bool)
                child[~ux] = p2[~ux]
                child1[~ux] = p1[~ux]
            
            #Mutation
            masked_indices = np.where(child == 0)[0]
            selected_index = np.random.choice(masked_indices)
            selected_weight = np.random.choice(obj[0][2][0].flatten())
            child[selected_index] = selected_weight
            new_population.append(child)
            
            masked_indices = np.where(child1 == 0)[0]
            selected_index = np.random.choice(masked_indices)
            selected_weight = np.random.choice(obj[0][2][0].flatten())
            child1[selected_index] = selected_weight
            new_population.append(child1)
            
        new_population.append(population[tfitness.index(max(tfitness))])
        population = new_population.copy()
    tfitness = [fitness(model, w, x, y) for w in population]
    max_fit = max(tfitness)
    weights = population[tfitness.index(max_fit)]
    model.layers[0].set_weights([weights[:14*28].reshape((14, 28)), weights[14*28:420]])
    model.layers[1].set_weights([weights[420:-1].reshape((28,1)), weights[-1].reshape((1,))])
    return model, max_fit

#NN Model
def baseline_model():
    input_shape = [x_train.shape[1]]
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(28, input_shape=input_shape, activation='sigmoid',
                            kernel_initializer=tf.keras.initializers.RandomUniform(0., 1.)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Training
model = baseline_model()
avg_fit = 0
num_runs = 1
for _ in range(num_runs):
    history, fit = genetic_algo(model, fitness_func, population_size=784, x=x_train, y=y_train, max_limit=19)
    avg_fit+=fit

print(f"Average fitness: {avg_fit/num_runs:.4f}")