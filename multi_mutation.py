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

# Fitness function to evaluate model performance
# Returns accuracy score by comparing predicted vs actual values
def fitness_func(model, x, y):
    y_pred = np.round(model.predict(x)).flatten()  # Flatten predictions to 1D array
    y = np.array(y)  # Convert y to numpy array
    return np.mean(y_pred == y)

# Evolutionary Strategy (μ,λ) implementation with multiple mutation rates
# Parameters:
# - model: base neural network model to evolve
# - fitness: fitness function to evaluate models
# - population_size: number of parents (μ)
# - num_generations: number of offspring per generation (λ)
# - sigmas: list of mutation rates for each weight layer
# - x, y: training data and labels
# - max_limit: maximum number of evolution cycles
# - sample_freq: frequency of sampling for adaptation
def evol_algo(model, fitness, population_size, num_generations, sigmas, x, y, max_limit, sample_freq):
    # Initialize population with random weights
    population = [tf.keras.models.clone_model(model) for i in range(population_size)]
    for i in range(population_size-1):
        population[i].set_weights([np.random.randn(*w.shape) for w in model.get_weights()])
    
    # Main evolution loop
    for _ in range(max_limit):
        counter = 0  # Track number of successful mutations
        for n in range(sample_freq):
            offsprings = []
            # Generate offspring through mutation
            for generation in range(num_generations):
                parent = random.choice(population)
                tempoff = tf.keras.models.clone_model(parent)
                # Add random noise to weights using layer-specific mutation rates
                new_weights = [w + sigma * np.random.randn(*w.shape) 
                             for w, sigma in zip(parent.get_weights(), sigmas)]
                tempoff.set_weights(new_weights)
                offsprings.append(tempoff)
                # Compare offspring fitness with parent
                fop = fitness(parent, x, y)
                fof = fitness(tempoff, x, y) 
                print(fop, fof)
                if fof>fop :
                    counter+=1
            # Replace population with offspring
            population = offsprings.copy()
            # Evaluate fitness of all individuals
            tfitness = np.array([fitness(m, x, y) for m in population])
            # Sort by fitness and select top performers
            sorted_indices = np.argsort(tfitness)
            population = [population[i] for i in sorted_indices[-population_size:]]
    # Return best performing model
    return population[-1]

#Evolutionary Strategy(myu+lambda)
def evolplusalgo(model, fitness, population_size, num_generations, sigmas, x, y, max_limit, sample_freq):
    """
    Evolutionary Strategy (μ+λ) implementation with multiple mutation rates
    Parents compete with offspring for survival, unlike (μ,λ) where parents are discarded
    
    Parameters:
    - model: base neural network model to evolve
    - fitness: fitness function to evaluate models
    - population_size: number of parents (μ) 
    - num_generations: number of offspring per generation (λ)
    - sigmas: list of mutation rates for each weight layer
    - x, y: training data and labels
    - max_limit: maximum number of evolution cycles
    - sample_freq: frequency of sampling for adaptation
    """
    # Initialize population with random weights
    population = [tf.keras.models.clone_model(model) for _ in range(population_size)]
    for i in range(population_size-1):
        population[i].set_weights([np.random.randn(*w.shape) for w in model.get_weights()])
    
    # Continue evolution until max_limit reached or fitness threshold met
    while max_limit>0 and fitness(population[-1], x, y)<0.7:
        gp = []  # Store fitness values for plotting
        
        # Generate and evaluate multiple generations
        for n in range(sample_freq):
            offsprings = []
            # Create offspring through mutation
            for generation in range(num_generations):
                parent = random.choice(population)
                tempoff = tf.keras.models.clone_model(parent)
                # Apply layer-specific mutations
                tempoff.set_weights([w + sigma * np.random.randn(*w.shape) for w, sigma in zip(parent.get_weights(), sigmas)])
                offsprings.append(tempoff)
            
            # Combine parents and offspring for selection
            population = population + offsprings
            # Evaluate fitness of all individuals
            tfitness = np.array([fitness(m, x, y) for m in population])
            # Select top performers based on fitness
            sorted_indices = np.argsort(tfitness)
            population = [population[i] for i in sorted_indices[-population_size:]]
            # Record best fitness for plotting
            gp.append(mean(tfitness))
        max_limit-=1
    
    # Visualize fitness progression
    plt.plot(gp)
    plt.title('Model Fitness')
    plt.ylabel('Fitness')
    plt.show()
    
    # Return best performing model
    return population[-1]

# Define neural network architecture
def baseline_model():
    input_shape = [x_train.shape[1]]
    model = tf.keras.Sequential([
        # Hidden layer with 28 neurons and sigmoid activation
        tf.keras.layers.Dense(
            units=28, 
            input_shape=input_shape,
            activation='sigmoid',
            kernel_initializer=tf.keras.initializers.RandomUniform(minval=0., maxval=1.)
        ),
        # Output layer with sigmoid activation for binary classification
        tf.keras.layers.Dense(units=1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Training loop with multiple trials
model = baseline_model()
trials = 1
avg_fit = 0

for _ in range(trials):
    # Generate random mutation rates for each weight layer
    sigmas = [np.round(np.random.rand(*w.shape), 1) for w in model.get_weights()]
    # Choose the algorithm to use, either evol_algo (μ,λ) or evolplusalgo (μ+λ). Currently using (μ+λ)
    #history = evol_algo(model, fitness_func, population_size=4, num_generations=20, 
    #                   sigmas=sigmas, x=x_train, y=y_train, max_limit=10, sample_freq=5)
    history = evolplusalgo(model, fitness_func, population_size=4, num_generations=20, 
                       sigmas=sigmas, x=x_train, y=y_train, max_limit=10, sample_freq=5)
    avg_fit += fitness_func(history, x_test, y_test)


# Visualize results using confusion matrix
typred = history.predict(x_test)
y_pred = np.round(typred)
con_mat = sklearn.metrics.confusion_matrix(y_test, y_pred)
plt.imshow(con_mat, cmap='Pastel1')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks([0, 1], ['Negative', 'Positive'])
plt.yticks([0, 1], ['False', 'True'])
plt.colorbar()
for i in range(2):
    for j in range(2):
        plt.text(j, i, con_mat[i, j], ha='center', va='center')
plt.show()