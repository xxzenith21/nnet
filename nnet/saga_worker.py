from PyQt5.QtCore import QObject, pyqtSignal
import hybrid_saga  # Import your Hybrid SAGA module

class HybridSAGAWorker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(int)
    result = pyqtSignal(object)  # Emit results

    def __init__(self, audio_files, conv_model_path, fc_model_path, label_mapping_file):
        super().__init__()
        self.audio_files = audio_files
        self.conv_model_path = conv_model_path
        self.fc_model_path = fc_model_path
        self.label_mapping_file = label_mapping_file

    def run(self):
        # Load models and label mapping from hybrid_saga
        conv_model, fc_model = hybrid_saga.load_models(self.conv_model_path, self.fc_model_path)
        label_mapping = hybrid_saga.load_label_mapping(self.label_mapping_file)

        for i, file_path in enumerate(self.audio_files):
            self.progress.emit(int((i / len(self.audio_files)) * 100))

            sound, sr = hybrid_saga.load_sound(file_path)  # Load the sound file
            processed_sound = hybrid_saga.process_sound(sound, sr)  # Process the sound file

            # Use your hybrid saga methods
            pseudo_labels = hybrid_saga.apply_hybrid_saga_method(processed_sound, conv_model, fc_model)

            # Emit results for this sound file
            textual_labels = hybrid_saga.convert_to_textual_labels(pseudo_labels, label_mapping)
            self.result.emit((file_path, textual_labels))

        self.finished.emit()

    