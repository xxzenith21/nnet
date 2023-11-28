import numpy as np
import os
import librosa
import shutil
import hashlib
from scipy.signal import convolve2d

# Genetic Algorithm (GA) Phase

# Step 1: Generate the initial k1 populations and initialize GA parameters
def initialize_population(k1, num_labels_list):
    population = []
    for num_labels in num_labels_list:
        min_label_index = 0
        max_label_index = 95
        individual = np.random.randint(min_label_index, max_label_index + 1, size=num_labels)
        population.append(individual)
    return population

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
def fitness_function(chromosome):
    # Modify this function according to your problem
    return -np.sum(chromosome**2)  # Example: minimize the sum of squares

# def roulette_wheel_selection(population, fitness_values):
#     # Handling the case where all fitness values are zero or negative
#     total_fitness = np.sum(fitness_values)
#     if total_fitness <= 0:
#         total_fitness = len(fitness_values)

#     # Normalizing selection probabilities
#     selection_probabilities = fitness_values / total_fitness

#     # Ensuring probabilities sum to 1
#     selection_probabilities = selection_probabilities / np.sum(selection_probabilities)

#     # Selecting indices based on the roulette wheel approach
#     selected_indices = np.random.choice(len(population), size=2, p=selection_probabilities)

#     # Using a list comprehension to select the individuals from the population
#     selected_individuals = [population[idx] for idx in selected_indices]
#     return selected_individuals

def tournament_selection(population, fitness_values, tournament_size):
    population_size = len(population)
    # Ensure tournament_size does not exceed population_size
    actual_tournament_size = min(tournament_size, population_size)

    # Convert population to a NumPy array if it's not already
    population_array = np.array(population)

    # Randomly select individuals for the tournament
    tournament_indices = np.random.choice(population_size, actual_tournament_size, replace=False)
    tournament_individuals = population_array[tournament_indices]
    tournament_fitness = fitness_values[tournament_indices]

    # Find the index of the best individual in the tournament
    best_index = np.argmax(tournament_fitness)
    best_individual = tournament_individuals[best_index]

    return best_individual

def crossover(parents):
    # Arithmetic crossover example
    alpha = np.random.rand()
    child1 = alpha * parents[0] + (1 - alpha) * parents[1]
    child2 = alpha * parents[1] + (1 - alpha) * parents[0]
    return child1, child2

# Placeholder: Step 5: Mutation
def mutate(chromosome, mutation_rate=0.1):
    # Gaussian mutation
    if np.random.rand() < mutation_rate:
        mutation_value = np.random.normal()
        gene_index = np.random.randint(len(chromosome))
        chromosome[gene_index] += mutation_value
    return chromosome

# Main GA Procedure
def run_genetic_algorithm(generations, population_size, population, crossover_rate, mutation_rate, stopping_generations):
    population = initialize_population(population_size, [num_genes])
    best_fitness = float('inf')

    for generation in range(generations):
        new_population = []
        fitness_values = np.array([fitness_function(individual) for individual in population])

        # Create new population
        for _ in range(population_size // 2):
            parent1 = tournament_selection(population, fitness_values, tournament_size=3)  # Example tournament size of 3
            parent2 = tournament_selection(population, fitness_values, tournament_size=3)

            offspring = crossover([parent1, parent2])
            new_population.extend([mutate(child) for child in offspring])
        
        population = new_population
    
    # Calculate final fitness values
    final_fitness_values = np.array([fitness_function(individual) for individual in population])

    return population, final_fitness_values

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

def simulated_annealing(chromosomes, sa_parameters):
    temperature, cooling_rate = sa_parameters
    current_solution = np.array(chromosomes[0])  # Convert to NumPy array

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
    final_population, fitness_values = run_genetic_algorithm(initial_population, generations, crossover_rate, mutation_rate, stopping_generations)
    best_chromosome = final_population[np.argmin(fitness_values)]

    # Phase 2: Simulated Annealing (SA)
    sa_parameters = initialize_sa_parameters()
    final_solution = simulated_annealing([best_chromosome], sa_parameters)


    for i in range(min(len(best_chromosome), len(unlabeled_sounds))):
        chromosome = best_chromosome[i]
        predicted_labels = predict_labels_using_models(chromosome, unlabeled_sounds[i], conv_weights, conv_bias, fc_weights, fc_bias)
        
        optimal_solution = simulated_annealing(predicted_labels, sa_parameters)
        final_solution.append([optimal_solution])  # Wrap the optimal solution in a list

    # Convert numeric output to pseudo-labels
    # pseudo_labels = [convert_to_labels(label_indices, index_to_label_mapping) for label_indices in final_labels]

    print("Final labels from SA:", final_solution)
    print("Index to Label Mapping:", index_to_label_mapping)
    print("Predicted Labels before SA:", predicted_labels)

    optimal_solution = simulated_annealing(predicted_labels, sa_parameters)
    print("Optimal solution from SA:", optimal_solution)
    
    # Return pseudo-labels
    return final_solution

# def convert_to_labels(indices, mapping):
#     if np.isscalar(indices):
#         # Handle a single value
#         return mapping.get(int(indices), "Unknown Label")
#     else:
#         # Handle an array of indices
#         return [mapping.get(int(index), "Unknown Label") for index in indices]

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

def load_label_mapping(mapping_file):
    label_mapping = np.load(mapping_file, allow_pickle=True).item()
    return label_mapping

def convert_indices_to_labels(indices, mapping):
    return [mapping.get(index, "Unknown Label") for index in indices]

# Assuming 'final_solution' is the output from your Simulated Annealing algorithm
def process_final_solution(final_solution, mapping):
    return convert_indices_to_labels(final_solution, mapping)

def normalize_and_discretize(solution, min_val, max_val, total_labels_count):
    # Normalize values to a 0-1 range
    normalized = (solution - min_val) / (max_val - min_val)

    # Scale to label index range and convert to integers
    discretized = (normalized * (total_labels_count - 1)).astype(int)

    return discretized

def analyze_sound_file(sound, sr=22050):
    # Extract MFCCs from the sound
    mfccs = librosa.feature.mfcc(y=sound, sr=sr, n_mfcc=13)

    # Calculate the mean of the first MFCC (as a simple feature)
    mean_mfcc = np.mean(mfccs[0, :])

    # Heuristic: Map the mean MFCC to a label count
    # This is a very basic example. You should develop a more sophisticated method.
    if mean_mfcc < -250:
        num_labels = 5
    elif -250 <= mean_mfcc < -150:
        num_labels = 8
    elif -150 <= mean_mfcc < -50:
        num_labels = 10
    else:
        num_labels = 12

    return num_labels 

mapping_file = 'K:/Thesis/labelMapping/label_to_index.npy'  
index_to_label_mapping = load_label_mapping(mapping_file)

# Example usage with specified parameters
conv_model_path = "K:/Thesis/models/conv_model.npz"
fc_model_path = "K:/Thesis/models/fc_model.npz"
unlabeled_sounds = "K:/Thesis/unlabeled_dataset"
# unlabeled_sounds = load_unlabeled_sounds(unlabeled_sounds)

saga_dataset = "K:/Thesis/saga_unlabeled_dataset"

k1 = 50
k2 = 10

generations = 100
sa_iterations = 50
crossover_rate = 0.8
mutation_rate = 0.05
stopping_generations = 50

population_size = 100
num_genes = 10
generations = 50

low = -5 
high = 5

# Get a list of all sound files in the folder
sound_files = [file for file in os.listdir(unlabeled_sounds) if file.endswith(".wav")]

# Initialize an empty list to store pseudo labels for all sound files
all_pseudo_labels = []

# Function to clear the contents of a folder
def clear_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

clear_folder(saga_dataset)

def extract_labels_from_filename(filename):
    # Check if the filename contains the pattern '. ' to split on
    if '. ' in filename:
        labels = filename.split('. ')[1]
        return labels.split(', ')
    else:
        # Return an empty list or placeholder if the pattern is not found
        return []

# Example usage
ground_truth_labels_dict = {}
for sound_file in sound_files:
    # Extracting file number
    file_number = sound_file.split('.')[0]

    # Removing '.wav' from the filename and then everything after the first period
    original_filename = sound_file.replace('.wav', '').split('.', 1)[1].strip()

    # Storing in dictionary
    ground_truth_labels_dict[file_number] = original_filename  # Storing the original filename

# Iterate over each sound file
for sound_file in sound_files:
    # Load the audio data from the file (you may need to adjust this based on your actual audio loading code)
    sound, _ = librosa.load(os.path.join(unlabeled_sounds, sound_file), sr=None)

    # Analyze the sound file to determine the number of labels
    num_labels = analyze_sound_file(sound)

    population_size = 100  # Example population size
    population = initialize_population(population_size, [num_labels] * population_size)
    final_population, fitness_values = run_genetic_algorithm(generations, population_size, population, crossover_rate, mutation_rate, stopping_generations)
    # Run Genetic Algorithm
    # population, fitness_values = run_genetic_algorithm(generations, population_size, num_genes)
    best_chromosome_index = np.argmin(fitness_values)
    best_chromosome = population[best_chromosome_index]

    # Initialize SA parameters
    sa_parameters = initialize_sa_parameters()

    # Apply Simulated Annealing on the best solution from GA
    final_solution = simulated_annealing([best_chromosome], sa_parameters)
    # print("Best solution from GA:", best_chromosome)
    # print("Final solution after SA:", final_solution)

    min_val = np.min(final_solution)
    max_val = np.max(final_solution)
    total_labels_count = len(index_to_label_mapping)

    # Normalize and discretize the final solution
    discrete_solution = normalize_and_discretize(final_solution, min_val, max_val, total_labels_count)

    # Convert discrete indices to labels
    textual_labels = [index_to_label_mapping.get(index, "Unknown Label") for index in discrete_solution]

    for number in discrete_solution:
        predicted_labels = [next((label for label, index in index_to_label_mapping.items() if index == number), "Unknown Label") for number in discrete_solution]

    # print("File Name:", sound_file)
    # print("Pseudo Labels:", predicted_labels)

    # Remove the "ARTURIA_sample_" prefix and clean labels
    # Clean each label individually
    cleaned_labels = [label.strip("[]'") for label in predicted_labels]
    file_number, _, labels = sound_file.partition(". ")
    
    # Create new file name
    # new_file_name = f"{os.path.splitext(sound_file)[0]}. {', '.join(cleaned_labels)}.wav"
    new_file_name = f"{file_number}. {', '.join(cleaned_labels)}.wav"

    # # Copy and rename the file to the new folder
    shutil.copy(os.path.join(unlabeled_sounds, sound_file), os.path.join(saga_dataset, new_file_name))

    # # Specify the part you want to remove from the filenames
    # part_to_remove = "ARTURIA_sample_"

    # # Iterate over all files in the folder
    # for filename in os.listdir(saga_dataset):
    #     if os.path.isfile(os.path.join(saga_dataset, filename)):
    #         # Check if the file is a regular file (not a directory)
    #         new_filename = filename.replace(part_to_remove, "")
        
    #        # Rename the file with the part removed
    #         os.rename(os.path.join(saga_dataset, filename), os.path.join(saga_dataset, new_filename))
    #         # print(f"Renamed: {filename} -> {new_filename}")

pseudo_labels_dict = {}
for filename in os.listdir(saga_dataset):  # Make sure saga_dataset is correctly defined in your environment
    if filename.endswith('.wav'):
        # Extracting file number
        file_number = filename.split('.')[0]

        # Extracting labels and removing '.wav' from the filename, then everything after the first period
        labels_part = filename.replace('.wav', '').split('.', 1)[1].strip()
        labels = [label.strip() for label in labels_part.split(',')]

        # Storing in dictionary
        pseudo_labels_dict[file_number] = labels_part

def calculate_accuracy_precision(ground_truth_dict, pseudo_labels_dict):
    correct_label_count = 0
    total_predicted_label_count = 0
    total_relevant_label_count = 0
    precision_per_file = []

    for file_number, ground_truth_labels in ground_truth_dict.items():
        pseudo_labels = pseudo_labels_dict.get(file_number, [])

        # Ensure labels are lists of strings
        ground_truth_labels = ground_truth_labels.split(', ')
        pseudo_labels = pseudo_labels.split(', ')

        correct_labels = set(ground_truth_labels).intersection(pseudo_labels)
        correct_label_count += len(correct_labels)
        total_relevant_label_count += len(ground_truth_labels)
        total_predicted_label_count += len(pseudo_labels)

        if len(pseudo_labels) > 0:
            precision = len(correct_labels) / len(pseudo_labels)
            precision_per_file.append(precision)

        print(f"File: {file_number}, Ground Truth: {ground_truth_labels}, Pseudo: {pseudo_labels}, Correct: {correct_labels}")

    accuracy = correct_label_count / total_relevant_label_count if total_relevant_label_count > 0 else 0
    average_precision = sum(precision_per_file) / len(precision_per_file) if precision_per_file else 0

    return accuracy, average_precision

# print(ground_truth_labels_dict, "\n\n\n")
# print(pseudo_labels_dict, "\n\n\n")

accuracy, precision = calculate_accuracy_precision(ground_truth_labels_dict, pseudo_labels_dict)
print("Accuracy:", accuracy)
print("Precision:", precision)
    


