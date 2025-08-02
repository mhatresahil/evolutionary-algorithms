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
	return data

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
# Returns accuracy score on given data
def fitness_func(model, x, y):
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    score = model.evaluate(x, y, verbose=1)
    return score[1]


# Evolutionary Strategy (μ,λ) implementation
# μ = population_size (number of parents)
# λ = num_generations (number of offspring per generation)
def evol_algo(model, fitness, population_size, num_generations, sigma, x, y, max_limit, sample_freq, r):
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
                # Add random noise to weights (mutation)
                tempoff.set_weights([w + sigma * np.random.randn(*w.shape) for w in tempoff.get_weights()])
                offsprings.append(tempoff)
                # Compare offspring fitness with parent
                fop = fitness(parent, x, y)
                fof = fitness(tempoff, x, y) 
                if fof>fop :
                    counter+=1
            # Replace population with offspring
            population = offsprings.copy()
            # Sort by fitness and select top performers
            tfitness = np.array([fitness(m, x, y) for m in population])
            sorted_indices = np.argsort(tfitness)
            population = [population[i] for i in sorted_indices[-population_size:]]
        
        # Adapt mutation step size (sigma)
        # Adjust mutation rate based on the proportion of successful
        # mutations over all attempted mutations in this cycle.
        total_mutations = sample_freq * num_generations
        if counter < total_mutations//5:
            sigma = (1-r)*sigma  # Decrease if few successful mutations
        else:
            sigma = (1+r)*sigma  # Increase if many successful mutations
    return population[-1]

# Evolutionary Strategy (μ+λ) implementation
# Different from (μ,λ) as parents compete with offspring for survival
def evolplusalgo(model, fitness, population_size, num_generations, sigma, x, y, max_limit, sample_freq, r):
    # Initialize population with random weights
    population = [tf.keras.models.clone_model(model) for i in range(population_size)]
    for i in range(population_size-1):
        population[i].set_weights([np.random.randn(*w.shape) for w in model.get_weights()])
    
    while max_limit>0:
        counter = 0  # Track successful mutations
        gp = []  # Store best fitness per generation for plotting
        for n in range(sample_freq):
            offsprings = []
            # Generate offspring through mutation
            for generation in range(num_generations):
                parent = random.choice(population)
                tempoff = tf.keras.models.clone_model(parent)
                # Add random noise to weights (mutation)
                tempoff.set_weights([w + sigma * np.random.randn(*w.shape) for w in tempoff.get_weights()])
                offsprings.append(tempoff)
                # Compare offspring fitness with parent
                fop = fitness(parent, x, y)
                fof = fitness(tempoff, x, y) 
                if fof>fop :
                    counter+=1
            # Combine parents and offspring
            population = population + offsprings
            # Sort by fitness and select top performers
            tfitness = np.array([fitness(m, x, y) for m in population])
            sorted_indices = np.argsort(tfitness)
            population = [population[i] for i in sorted_indices[-population_size:]]
            gp.append(fitness(population[-1], x, y))
        
        # Adapt mutation step size (sigma)
        # Adjust mutation rate based on overall success rate in this cycle
        total_mutations = sample_freq * num_generations
        if counter < total_mutations//5:
            sigma = (1-r)*sigma  # Decrease if few successful mutations
        else:
            sigma = (1+r)*sigma  # Increase if many successful mutations
        max_limit-=1
    
    # Plot fitness progression
    plt.plot(gp)
    plt.title('model fitness')
    plt.ylabel('fitness')
    plt.show()
    return population[-1]

# Define neural network architecture
def baseline_model():
    input_shape = [x_train.shape[1]]
    initializer = tf.keras.initializers.RandomUniform(minval=0., maxval=1.)
    model = tf.keras.Sequential()
    # Hidden layer with 28 neurons and sigmoid activation
    model.add(tf.keras.layers.Dense(units = 28, input_shape = input_shape, activation = 'sigmoid', kernel_initializer = initializer))
    # Output layer with sigmoid activation for binary classification
    model.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return model

# Train model using evolutionary strategy
model = baseline_model()
avg_fit = 0
for _ in range(1):
    # Choose between (μ+λ) and (μ,λ) optimization based on the desired behavior
    # Currently using (μ,λ)
    #history = evolplusalgo(model, fitness_func, population_size=4, num_generations=20, sigma=0.5, x=x_train, y=y_train, max_limit=20, sample_freq=10, r = 0.2)
    history = evol_algo(model, fitness_func, population_size=4, num_generations=20, sigma=0.5, x=x_train, y=y_train, max_limit=20, sample_freq=10, r = 0.2)
    avg_fit+=fitness_func(history, x_test, y_test)                                                                                              
    
print(avg_fit/10)

# Generate and plot confusion matrix
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