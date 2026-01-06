import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from vame.util.auxiliary import read_config
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QSpinBox, QPushButton, QFileDialog, QMessageBox, QComboBox)

class ReconstructionGUI(QWidget):
    def __init__(self, cfg, columns_names):
        super().__init__()
        self.cfg = cfg
        self.columns_names = columns_names
        self.default_path = os.path.join(self.cfg['project_path'], 'data', 'train')
        self.reconstruct_path = os.path.join(self.cfg['project_path'], 'model', 'evaluate')
        self.original_data = None
        self.reconstructed_data = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Feature Reconstruction Plotter')
        self.setGeometry(100, 100, 500, 200)
        
        layout = QVBoxLayout()

        # Original Data
        self.btn_load_orig = QPushButton('Select Original Data (.npy)')
        self.btn_load_orig.clicked.connect(self.load_original)
        layout.addWidget(self.btn_load_orig)
        self.lbl_orig = QLabel('No file selected')
        layout.addWidget(self.lbl_orig)

        # Reconstructed Data
        self.btn_load_recon = QPushButton('Select Reconstructed Data (.npy)')
        self.btn_load_recon.clicked.connect(self.load_reconstructed)
        layout.addWidget(self.btn_load_recon)
        self.lbl_recon = QLabel('No file selected')
        layout.addWidget(self.lbl_recon)

        # Feature Selector
        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel('Feature:'))
        self.combo_feature = QComboBox()
        self.combo_feature.addItems(self.columns_names)
        h_layout.addWidget(self.combo_feature)
        layout.addLayout(h_layout)

        # Plot Button
        self.btn_plot = QPushButton('Plot')
        self.btn_plot.clicked.connect(self.plot_data)
        layout.addWidget(self.btn_plot)

        self.setLayout(layout)

    def load_original(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open Original Data', self.default_path, 'Numpy Files (*.npy)')
        if fname:
            try:
                self.original_data = np.load(fname)
                self.lbl_orig.setText(os.path.basename(fname))
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def load_reconstructed(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open Reconstructed Data', self.reconstruct_path, 'Numpy Files (*.npy)')
        if fname:
            try:
                self.reconstructed_data = np.load(fname)
                self.lbl_recon.setText(os.path.basename(fname))
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def plot_data(self):
        if self.original_data is None or self.reconstructed_data is None:
            QMessageBox.warning(self, "Warning", "Please load both original and reconstructed data files.")
            return

        idx = self.combo_feature.currentIndex()
        
        try:
            orig = self.original_data
            recon = self.reconstructed_data
            
            # Select feature and flatten
            y_orig = orig[idx,:].flatten()
            y_recon = recon[idx,].flatten()

            # Ensure same length
            min_len = min(len(y_orig), len(y_recon))
            y_orig = y_orig[:min_len]
            y_recon = y_recon[:min_len]

            plt.figure(figsize=(20, 5))
            plt.plot(y_orig, label='Original', color ='k', alpha=0.5, linewidth=1)
            plt.plot(y_recon,'r--', label='Reconstruction', alpha=0.5, linewidth=1)
            
            plt.title(f'Reconstruction of Feature {self.combo_feature.currentText()}')
            plt.xlabel('Time steps')
            plt.ylabel('Value')
            plt.legend()
            plt.tight_layout()
            plt.show()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error plotting data: {e}")

class plot_feature_reconstruction:
    def __init__(self, config):
        config_file = Path(config).resolve()
        self.cfg = read_config(config_file)
        self.path_to_file = self.cfg['project_path']
        self.filename = self.cfg['video_sets']
    
        


    def __call__(self):
        app = QApplication.instance()
        columns_names = self.extract_landmarks_names()
        if not app:
            app = QApplication(sys.argv)
            self.gui = ReconstructionGUI(self.cfg, columns_names)
            self.gui.show()
            app.exec_()
        else:
            self.gui = ReconstructionGUI(self.cfg, columns_names)
            self.gui.show()

    def extract_landmarks_names(self):
        fname = self.filename[0] if isinstance(self.filename, list) else self.filename
        header = pd.read_csv(os.path.join(self.path_to_file,"videos","pose_estimation",fname+'.csv'), nrows=0)
        columns_names = header.columns.tolist()[1:]  # Skip the first column which is usually an index or frame number
        #remove the columns which its name contains the name score
        columns_names = [name for name in columns_names if 'score' not in name.lower()]
        return columns_names