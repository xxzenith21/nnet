import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QTextEdit, QProgressBar, QTabWidget, QWidget
from PyQt5.QtCore import QTimer, QThread, QObject, pyqtSignal
from saga_worker import HybridSAGAWorker
import hybrid_saga as hsa
import librosa
import numpy as np
from hybrid_saga import hybrid_saga

class PseudoLabelingWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pseudolabeling & Synthesizer Prediction Interface")
        self.setGeometry(100, 100, 800, 600)
        
        self.initUI()

    def initUI(self):
        self.load_button = QPushButton("Load Audio File(s)", self)
        self.load_button.setGeometry(50, 50, 200, 30)
        self.load_button.clicked.connect(self.load_audio)

        self.startButton = QPushButton("Start Processing", self)
        self.startButton.setGeometry(50, 100, 200, 30)
        self.startButton.clicked.connect(self.startProcessing)
        self.startButton.setEnabled(False)  # Disable until files are loaded

        self.progressBar = QProgressBar(self)
        self.progressBar.setGeometry(50, 150, 700, 30)

        self.pseudolabels_display = QTextEdit(self)
        self.pseudolabels_display.setGeometry(50, 200, 700, 300)
        self.pseudolabels_display.setReadOnly(True)

        self.loaded_files = []

    def load_audio(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        options |= QFileDialog.ExistingFiles
        options |= QFileDialog.Directory
        self.loaded_files, _ = QFileDialog.getOpenFileNames(self, "Open Audio File(s) or Folder", "", "Audio Files (*.wav *.mp3);;All Files (*)", options=options)
        if self.loaded_files:
            self.startButton.setEnabled(True)

    def startProcessing(self):
        self.thread = QThread()
        self.worker = HybridSAGAWorker(self.loaded_files)
        self.worker.moveToThread(self.thread)
        self.worker.finished.connect(self.thread.quit)
        self.worker.progress.connect(self.progressBar.setValue)
        self.worker.result.connect(self.displayResults)
        self.thread.started.connect(self.worker.run)
        self.thread.start()
        self.startButton.setEnabled(False)

    def displayResults(self, results):
        self.pseudolabels_display.clear()
        for result in results:
            self.pseudolabels_display.append(', '.join(result))

    def startHybridSAGA(self):
    # Load models and mappings
        conv_weights, conv_bias, fc_weights, fc_bias = self.worker.load_models()
        index_to_label_mapping = self.worker.load_label_mapping()

        results = []

        # Process each audio file
        for i, audio_file in enumerate(self.loaded_files):
            # Update progress
            self.progressBar.setValue(int((i / len(self.loaded_files)) * 100))

            # Load the audio file
            sound, _ = librosa.load(audio_file, sr=None)

            # Call your hybrid saga algorithm with the sound and models
            pseudo_labels = self.worker.apply_hybrid_saga(sound, conv_weights, conv_bias, fc_weights, fc_bias, index_to_label_mapping)

            # Convert numeric labels to textual labels using the mapping file
            textual_labels = self.worker.convert_to_textual_labels(pseudo_labels)

            results.append(textual_labels)

        # Emit results for all audio files
        self.displayResults(results)

        # Signal the completion of the process
        self.worker.finished.emit()


class HybridSAGAWorker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(int)
    result = pyqtSignal(list)

    def __init__(self, audio_files):
        super().__init__()
        self.audio_files = audio_files
        # Paths to your model and mapping files
        self.conv_model_path = "K:/Thesis/models/conv_model.npz"
        self.fc_model_path = "K:/Thesis/models/fc_model.npz"
        self.label_mapping_file = "K:/Thesis/labelMapping/label_to_index.npy"

    def run(self):
        # Load models and mappings
        conv_weights, conv_bias, fc_weights, fc_bias = self.load_models()
        index_to_label_mapping = self.load_label_mapping()

        results = []

        # Process each audio file
        for i, audio_file in enumerate(self.audio_files):
            # Update progress
            self.progress.emit(int((i / len(self.audio_files)) * 100))

            # Load the audio file
            sound, _ = librosa.load(audio_file, sr=None)

            # Call your hybrid saga algorithm with the sound and models
            pseudo_labels = self.apply_hybrid_saga(sound, conv_weights, conv_bias, fc_weights, fc_bias, index_to_label_mapping)

            # Convert numeric labels to textual labels using the mapping file
            textual_labels = self.convert_to_textual_labels(pseudo_labels)

            results.append(textual_labels)

        # Emit results for all audio files
        self.result.emit(results)

        # Signal the completion of the process
        self.finished.emit()
    
    def initialize_population(self, k1, num_labels_list):
        # Initialize the population with random values
        population = []

        for _ in range(k1):
        # Generate a random chromosome with values between 0 and the number of labels for each feature
            chromosome = [np.random.randint(0, num_labels) for num_labels in num_labels_list]
            population.append(chromosome)

        return population

    def preprocess_audio(self, sound, sr):
        # Example of extracting MFCCs
        mfccs = librosa.feature.mfcc(y=sound, sr=sr, n_mfcc=20)
        return mfccs

    def load_models(self):
        conv_model_data = np.load(self.conv_model_path)
        fc_model_data = np.load(self.fc_model_path)

        conv_weights = conv_model_data['weights']
        conv_bias = conv_model_data['bias']
        fc_weights = fc_model_data['weights']
        fc_bias = fc_model_data['bias']

        return conv_weights, conv_bias, fc_weights, fc_bias
    
    def load_label_mapping(self):
        label_mapping = np.load(self.label_mapping_file, allow_pickle=True).item()
        return {v: k for k, v in label_mapping.items()}

    def generate_conv_features(self, audio, conv_weights, conv_bias):
        # Assuming audio data is a single 1D numpy array and conv_weights, conv_bias are your loaded model parameters
        # Adjust the shape of audio data and conv_weights if necessary
        conv_result = np.convolve(audio, conv_weights.flatten()) + conv_bias
        return conv_result

    def process_with_fc_model(self, features, fc_weights, fc_bias):
        # Assuming a simple dot product for the FC layer
        fc_output = np.dot(features, fc_weights) + fc_bias
        return fc_output

    def apply_hybrid_saga(self, sound, conv_weights, conv_bias, fc_weights, fc_bias, index_to_label_mapping):
        # Initialize parameters for GA and SA
        k1 = 50  # example value
        num_labels_list = [len(index_to_label_mapping)]  # Adjust as per your model output structure
        initial_population = self.initialize_population(k1, num_labels_list)

        # Run Genetic Algorithm
        generations = 500  # example value
        crossover_rate = 0.9  # example value
        mutation_rate = 0.1  # example value
        stopping_generations = 50  # example value
        final_population, fitness_values = self.run_genetic_algorithm(generations, k1, initial_population, crossover_rate, mutation_rate, stopping_generations)

        # Select best chromosome
        best_chromosome = final_population[np.argmin(fitness_values)]

        # Apply Simulated Annealing
        sa_parameters = self.initialize_sa_parameters()
        final_solution = self.simulated_annealing([best_chromosome], sa_parameters)

        # Convert the solution to discrete label indices
        # Assuming a method to discretize and convert to labels
        discrete_labels = self.discretize_solution(final_solution)

        return discrete_labels

    def convert_to_textual_labels(self, numeric_labels):
        return [self.index_to_label_mapping.get(index, "Unknown Label") for index in numeric_labels]

    
def main():
    app = QApplication(sys.argv)
    window = PseudoLabelingWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
