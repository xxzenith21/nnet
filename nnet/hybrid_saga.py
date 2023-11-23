import numpy as np
import os
import librosa
from scipy.signal import convolve2d

# Genetic Algorithm (GA) Phase

# Step 1: Generate the initial k1 populations and initialize GA parameters
def initialize_population(k1, num_labels):
    return np.random.randint(2, size=(k1, num_labels))

# Placeholder: Implement your crossover operator
def apply_crossover(population, crossover_rate):
    num_parents = population.shape[0]
    num_labels = population.shape[1]

    # Ensure an even number of parents for crossover
    num_parents_for_crossover = max(num_parents - (num_parents % 2), 2)

    # Determine the number of crossover pairs
    num_crossover_pairs = min(int(num_parents_for_crossover * crossover_rate), num_parents_for_crossover // 2)

    # Ensure at least one crossover pair
    num_crossover_pairs = max(num_crossover_pairs, 1)

    # Randomly select pairs of parents for crossover
    crossover_pairs = np.random.choice(num_parents_for_crossover, size=(num_crossover_pairs, 2), replace=False)

    for pair in crossover_pairs:
        # Randomly select the crossover point
        crossover_point = np.random.randint(1, num_labels - 1)

        # Perform one-point crossover
        temp = np.copy(population[pair[0], crossover_point:])
        population[pair[0], crossover_point:] = population[pair[1], crossover_point:]
        population[pair[1], crossover_point:] = temp

    return population


# Placeholder: Implement your mutation operator
def apply_mutation(population, mutation_rate):
    num_individuals = population.shape[0]
    num_labels = population.shape[1]

    # Determine the number of mutations
    num_mutations = int(num_individuals * num_labels * mutation_rate)

    # Randomly select positions for mutation
    mutation_positions = np.random.choice(num_individuals * num_labels, size=num_mutations, replace=False)

    # Perform bit-flip mutation at selected positions
    population_flat = population.flatten()
    for position in mutation_positions:
        population_flat[position] = 1 - population_flat[position]

    # Reshape the mutated flat array to the original population shape
    mutated_population = population_flat.reshape((num_individuals, num_labels))

    return mutated_population

# Placeholder: Implement your fitness function
def evaluate_population_fitness(population):
    # Placeholder logic for evaluating fitness of each individual in the population
    # In this example, fitness is the sum of components in each chromosome
    fitness_values = np.sum(population, axis=1)

    return fitness_values

# Step 3: Choose k good chromosomes for SA
def select_chromosomes_for_sa(population, k):
    num_chromosomes = population.shape[0]

    # Calculate pairwise Hamming distances
    hamming_distances = np.zeros((num_chromosomes, num_chromosomes))
    for i in range(num_chromosomes):
        for j in range(i + 1, num_chromosomes):
            hamming_distances[i, j] = np.sum(population[i] != population[j])
            hamming_distances[j, i] = hamming_distances[i, j]

    # Find the indices of the top k chromosomes with the smallest distances
    selected_indices = np.argsort(np.sum(hamming_distances, axis=0))[:k]

    # Select the corresponding chromosomes
    selected_chromosomes = population[selected_indices]

    return selected_chromosomes




# Step 2: Run GA for m generations
# Actual GA logic
def run_genetic_algorithm(population, generations, crossover_rate, mutation_rate, stopping_generations):
    best_fitness = -1  # Placeholder for the best fitness value
    no_improvement_count = 0

    for generation in range(generations):
        # Apply GA operators to evolve the population
        population = apply_crossover(population, crossover_rate)
        population = apply_mutation(population, mutation_rate)

        # Evaluate fitness
        fitness_values = evaluate_population_fitness(population)

        # Update best_fitness based on the maximum fitness value
        current_best_fitness = np.max(fitness_values)
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        # Check stopping criterion
        if no_improvement_count >= stopping_generations:
            break

    # Return the final population and its fitness values
    return population, fitness_values

# Step 3: Choose k good chromosomes for SA
def select_chromosomes_for_sa(population, k):
    num_chromosomes = population.shape[0]

    # Calculate pairwise Hamming distances
    hamming_distances = np.zeros((num_chromosomes, num_chromosomes))
    for i in range(num_chromosomes):
        for j in range(i+1, num_chromosomes):
            hamming_distances[i, j] = np.sum(population[i] != population[j])
            hamming_distances[j, i] = hamming_distances[i, j]

    # Find the indices of the top k chromosomes with the smallest distances
    selected_indices = np.argsort(np.sum(hamming_distances, axis=0))[:k]

    # Select the corresponding chromosomes
    selected_chromosomes = population[selected_indices]

    return selected_chromosomes

# Simulated Annealing (SA) Phase

# Step 4: Initialize SA parameters
def initialize_sa_parameters():
    # Define SA parameters (temperature, cooling schedule, etc.)
    temperature = 50.0
    cooling_rate = 0.95
    return temperature, cooling_rate

# Step 5: Apply SA algorithm
def simulated_annealing(chromosomes, sa_parameters):
    temperature, cooling_rate = sa_parameters
    current_solution = chromosomes[0]  # Start with the first chromosome

    while temperature > 0.1:  # Adjust stopping criterion as needed
        # Generate neighbor solution by perturbation
        neighbor_solution = perturb_solution(current_solution)

        # Evaluate neighbor solutions for their fitness
        current_fitness = evaluate_fitness(current_solution)
        neighbor_fitness = evaluate_fitness(neighbor_solution)

        # Accept neighbor solution if better than current solution
        if neighbor_fitness > current_fitness or acceptance_probability(neighbor_fitness - current_fitness, temperature):
            current_solution = neighbor_solution

        # Cooling
        temperature *= cooling_rate

    return current_solution

# Placeholder: Implement perturbation logic
def perturb_solution(solution):
    # Placeholder logic for perturbation
    perturbed_solution = np.copy(solution)  # Replace this with actual perturbation logic
    return perturbed_solution

# Placeholder: Implement acceptance probability logic (Simulated Annealing)
def acceptance_probability(energy_diff, temperature):
    if energy_diff < 0:
        return True
    return np.random.rand() < np.exp(-energy_diff / temperature)

# Placeholder: Implement your fitness function
def evaluate_fitness(solution):
    # Example logic: Fitness based on the diversity of labels in the solution
    # Assuming 'solution' is an array of label indices or some form of label representation

    # Calculate diversity as an example metric. This can be the number of unique labels, 
    # variance in label distribution, or any other measure that makes sense for your problem.
    unique_labels = np.unique(solution)
    diversity_score = len(unique_labels)  # More unique labels result in a higher score

    return diversity_score

# Main Hybrid SAGA Procedure

def load_unlabeled_sounds(directory_path):
    # Specify the path to the directory containing unlabeled sounds
    # Adjust this based on the actual location of your unlabeled sounds
    unlabeled_sounds = []

    for filename in os.listdir(directory_path):
        if filename.endswith(".wav"):
            file_path = os.path.join(directory_path, filename)
            sound, _ = librosa.load(file_path, sr=None)  # sr=None to preserve the original sampling rate
            unlabeled_sounds.append(sound)

    return unlabeled_sounds

# Sigmoid
def activation_function(result):
    # Replace this with your actual activation function (e.g., sigmoid, softmax, etc.)
    return 1 / (1 + np.exp(-result))

def load_models(conv_model_path, fc_model_path):
    try:
        conv_model_data = np.load(conv_model_path)
        fc_model_data = np.load(fc_model_path)

        # Print available keys
        print("Keys in Conv Model:", conv_model_data.files)
        print("Keys in FC Model:", fc_model_data.files)

        # Use the correct keys
        conv_weights = conv_model_data['weights']
        conv_bias = conv_model_data['bias']

        fc_weights = fc_model_data['weights']
        fc_bias = fc_model_data['bias']

        return conv_weights, conv_bias, fc_weights, fc_bias
    except Exception as e:
        print(f"Error loading models: {e}")
        # Add appropriate error handling or raise an exception based on your needs

def load_label_mapping(mapping_file):
    label_to_index_mapping = np.load(mapping_file, allow_pickle=True).item()
    # Invert the mapping to create an index-to-label mapping
    return {v: k for k, v in label_to_index_mapping.items()}

def hybrid_saga(conv_model_path, fc_model_path, unlabeled_sounds, k1, k2, num_labels, generations, sa_iterations, crossover_rate, mutation_rate, stopping_generations):
    # Load label mapping
    index_to_label_mapping = load_label_mapping('K:/Thesis/labelMapping/label_to_index.npy')
    
    # Load Conv2DLayer and FullyConnectedLayer models
    conv_weights, conv_bias, fc_weights, fc_bias = load_models(conv_model_path, fc_model_path)

    print("Shape of conv_weights:", conv_weights.shape)
    print("Shape of conv_bias:", conv_bias.shape)
    print("Shape of fc_weights:", fc_weights.shape)
    
    # Phase 1: Genetic Algorithm (GA)
    initial_population = initialize_population(k1, num_labels)
    final_population, _ = run_genetic_algorithm(initial_population, generations, crossover_rate, mutation_rate, stopping_generations)
    selected_chromosomes = select_chromosomes_for_sa(final_population, k2)

    # Phase 2: Simulated Annealing (SA)
    sa_parameters = initialize_sa_parameters()
    final_labels = []

    for i in range(min(len(selected_chromosomes), len(unlabeled_sounds))):
        chromosome = selected_chromosomes[i]
        predicted_labels = predict_labels_using_models(chromosome, unlabeled_sounds[i], conv_weights, conv_bias, fc_weights, fc_bias)
        
        optimal_solution = simulated_annealing(predicted_labels, sa_parameters)
        final_labels.append([optimal_solution])  # Wrap the optimal solution in a list

    # Convert numeric output to pseudo-labels
    pseudo_labels = [convert_to_labels(label_indices, index_to_label_mapping) for label_indices in final_labels]

    print("Final labels from SA:", final_labels)
    print("Index to Label Mapping:", index_to_label_mapping)
    print("Predicted Labels before SA:", predicted_labels)

    optimal_solution = simulated_annealing(predicted_labels, sa_parameters)
    print("Optimal solution from SA:", optimal_solution)
    
    # Return pseudo-labels
    return pseudo_labels



def convert_to_labels(indices, mapping):
    if np.isscalar(indices):
        # Handle a single value
        return mapping.get(int(indices), "Unknown Label")
    else:
        # Handle an array of indices
        return [mapping.get(int(index), "Unknown Label") for index in indices]

# Modify the predict_labels function to use both models
def predict_labels_using_models(chromosome, unlabeled_sound, conv_weights, conv_bias, fc_weights, fc_bias):
    # Ensure unlabeled_sound has the same shape as conv_weights
    unlabeled_sound = unlabeled_sound.reshape((len(unlabeled_sound), 1))

    # Get the dimensions of the convolutional weights
    num_filters, _, filter_rows, filter_cols = conv_weights.shape

    # Initialize the convolution result
    conv_result = np.zeros((unlabeled_sound.shape[0] - filter_rows + 1, num_filters))

    # Perform convolution operation
    for i in range(conv_result.shape[0]):
        for filter_index in range(num_filters):
            conv_result[i, filter_index] = np.sum(unlabeled_sound[i:i + filter_rows, 0] * conv_weights[filter_index, 0])

    # Add bias to each filter's result
    conv_bias_reshaped = conv_bias.flatten()
    for filter_index in range(num_filters):
        conv_result[:, filter_index] += conv_bias_reshaped[filter_index]

    # Sum over filters
    conv_result_summed = np.sum(conv_result, axis=1)

    # Expand chromosome to match the input size of fc_weights
    target_size = fc_weights.shape[0]  # Get the target size from fc_weights
    expanded_chromosome = np.zeros(target_size)
    expanded_chromosome[:len(chromosome)] = chromosome
    chromosome_row_vector = expanded_chromosome.reshape(1, -1)

    # Perform the dot product with expanded chromosome
    fc_result = np.dot(chromosome_row_vector, fc_weights) + fc_bias

    # Reshape conv_result_summed for concatenation and concatenate with fc_result
    conv_result_summed_reshaped = conv_result_summed.reshape(1, -1)
    combined_result = np.concatenate((conv_result_summed_reshaped, fc_result), axis=1)

    # Apply activation function if needed
    predicted_labels = activation_function(combined_result)

    # Apply activation function
    predicted_labels_continuous = activation_function(combined_result)

    # Convert continuous outputs to discrete label indices
    predicted_labels_indices = np.argmax(predicted_labels_continuous, axis=1)

    return predicted_labels_indices







# Example usage with specified parameters
conv_model_path = "K:/Thesis/models/conv_model.npz"
fc_model_path = "K:/Thesis/models/fc_model.npz"
unlabeled_sounds = "K:/Thesis/unlabeled_dataset"  # Replace with the actual path
unlabeled_sounds = load_unlabeled_sounds(unlabeled_sounds)
k1 = 50
k2 = 10
num_labels = 6  # Adjust based on the number of labels in your problem
generations = 100
sa_iterations = 50
crossover_rate = 0.8
mutation_rate = 0.05
stopping_generations = 50

result = hybrid_saga(conv_model_path, fc_model_path, unlabeled_sounds, k1, k2, num_labels, generations, sa_iterations, crossover_rate, mutation_rate, stopping_generations)
print(result)

