import numpy as np
import os
import librosa
import shutil
import hashlib
from scipy.signal import convolve2d

#import desc_learn

class GeneticAlgorithmOperations:
    def __init__(self, ground_truth_labels_dict):
        self.ground_truth_labels_dict = ground_truth_labels_dict

    def initialize_population(self, k1, num_labels_list):
        population = []

        for num_labels in num_labels_list:
            min_label_index = 0
            max_label_index = 95
            individual = np.random.randint(min_label_index, max_label_index + 1, size=num_labels)
            population.append(individual)

        return population

    def apply_crossover(self, population, crossover_rate):
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

    def apply_mutation(self, population, mutation_rate):
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
    
    def jaccard_similarity(set1, set2):
        
        intersection = len(set(set1).intersection(set2))
        union = len(set(set1).union(set2))
        return intersection / union if union != 0 else 0


    def evaluate_fitness(self, chromosome):
        # Assuming the first element of the chromosome is the file number
        file_number = chromosome[0]

        # Rest of the chromosome represents predicted labels
        predicted_labels = set(chromosome[1:])

        # Get ground truth labels for the file number and convert to set
        true_labels = set(ground_truth_labels.get(str(file_number), "").split(', '))

        # Calculate Jaccard similarity
        fitness_score = jaccard_similarity(predicted_labels, true_labels)

        return fitness_score

    # ... other GA-related methods ...

class SimulatedAnnealingOperations:
    def __init__(self, ground_truth_labels_dict):
        self.ground_truth_labels_dict = ground_truth_labels_dict

    def initialize_sa_parameters(self):
        # ... existing implementation ...

    def run_sa(self, chromosomes):
        # ... existing implementation ...

    def perturb_solution(self, solution, perturbation_rate):
        # ... existing implementation ...

    def acceptance_probability(self, energy_diff, temperature):
        # ... existing implementation ...

    # ... other SA-related methods ...

class ModelLoader:
    def __init__(self, conv_model_path, fc_model_path):
        self.conv_model_path = conv_model_path
        self.fc_model_path = fc_model_path

    def load_models(self):
        # ... existing implementation ...

class DataHandler:
    def __init__(self, unlabeled_sounds_path):
        self.unlabeled_sounds_path = unlabeled_sounds_path

    def load_unlabeled_sounds(self):
        # ... existing implementation ...

    # ... other data handling methods ...











    def main():
    # Load and preprocess data
    ground_truth_labels_dict = # ... load your ground truth labels ...
    data_handler = DataHandler(unlabeled_sounds_path)
    unlabeled_sounds = data_handler.load_unlabeled_sounds()

    # Initialize GA and SA operations
    ga_operations = GeneticAlgorithmOperations(ground_truth_labels_dict)
    sa_operations = SimulatedAnnealingOperations(ground_truth_labels_dict)

    # Load models
    model_loader = ModelLoader(conv_model_path, fc_model_path)
    conv_weights, conv_bias, fc_weights, fc_bias = model_loader.load_models()

    # Run the hybrid algorithm
    hybrid_ga_sa = HybridGeneticSimulatedAnnealing(ga_operations, sa_operations, model_loader, data_handler)
    hybrid_ga_sa.run()

if __name__ == "__main__":
    main()