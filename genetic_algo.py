import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
import sklearn.metrics
import random
from multiprocessing import Pool, cpu_count

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

def evaluate_individual(args):
    model, weights, x, y = args
    return fitness_func(model, weights, x, y)

#Genetic Algorithm
def genetic_algo(model, fitness, population_size, x, y, max_limit):
    """
    Implements a genetic algorithm to optimize neural network weights
    
    Args:
        model: Neural network model to optimize
        fitness: Fitness function to evaluate individuals
        population_size: Size of population in each generation
        x: Input features for fitness evaluation
        y: Target values for fitness evaluation
        max_limit: Maximum number of generations
        
    Returns:
        model: Model with optimized weights
        max_fit: Maximum fitness achieved
    """
    
    # Initialize population with masked weight vectors
    population = []
    for i in range(population_size):
        weights = model.get_weights()
        weights_vector = np.concatenate([w.flatten() for w in weights])
        # Apply mask with 10% probability of keeping weights
        mask = np.random.choice([0, 1], size=weights_vector.shape, p=[0.9, 0.1])
        population.append(weights_vector*mask)
    
    # Set up parallel processing
    num_cores = cpu_count()
    pool = Pool(processes=num_cores)
    
    # Main evolution loop
    for _ in range(max_limit):
        new_population = []
        
        # Calculate fitness scores for entire population in parallel
        fitness_args = [(model, w, x, y) for w in population]
        tfitness = pool.map(evaluate_individual, fitness_args)
        
        # Calculate selection probabilities based on fitness
        total_fit = sum(tfitness)
        probabilities = [score / total_fit for score in tfitness]
        
        # Generate new population through selection, crossover and mutation
        for j in range(population_size//2):
            # Select two parents based on fitness proportions
            p1_index = np.random.choice(len(tfitness), replace = False, p = probabilities)
            p2_index = np.random.choice(len(tfitness), replace = False, p = probabilities)
            p1 = population[p1_index]
            p2 = population[p2_index]
            
            # Create copies for offspring
            child = p1.copy()
            child1 = p2.copy()
            
            # Perform uniform crossover with 90% probability
            if random.choices([True, False], weights=[0.9, 0.1])[0]:
                par_shape = p1.shape
                ux = np.random.randint(low=0, high=2, size=par_shape).astype(bool)
                child[~ux] = p2[~ux]  # Swap genes where ux is False
                child1[~ux] = p1[~ux]
            
            # Mutate first child by randomly activating one masked weight
            masked_indices = np.where(child == 0)[0]
            if len(masked_indices) > 0:
                selected_index = np.random.choice(masked_indices)
                child[selected_index] = np.random.uniform(0, 1)
            new_population.append(child)
            
            # Mutate second child similarly
            masked_indices = np.where(child1 == 0)[0]
            if len(masked_indices) > 0:  # Only mutate if there are masked indices
                selected_index = np.random.choice(masked_indices)
                child1[selected_index] = np.random.uniform(0, 1)
            new_population.append(child1)
            
        # Elitism: Keep best individual from previous generation
        new_population.append(population[tfitness.index(max(tfitness))])
        population = new_population.copy()
    
    # Final fitness evaluation of last generation
    fitness_args = [(model, w, x, y) for w in population]
    tfitness = pool.map(evaluate_individual, fitness_args)
    
    # Clean up parallel processing
    pool.close()
    pool.join()
    
    # Get best performing individual and update model weights
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

if __name__ == '__main__':
    # Training
    model = baseline_model()
    avg_fit = 0
    num_runs = 1
    for _ in range(num_runs):
        history, fit = genetic_algo(model, fitness_func, population_size=784, x=x_train, y=y_train, max_limit=10)
        avg_fit+=fit

    print(f"Average fitness: {avg_fit/num_runs:.4f}")