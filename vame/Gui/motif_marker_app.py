import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from vame.util.auxiliary import read_config
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QSpinBox, QPushButton, QFileDialog, QMessageBox, QComboBox)

class MotifMarkGUI(QWidget):
    def __init__(self, cfg, columns_names):
        super().__init__()
        self.cfg = cfg
        self.columns_names = columns_names
        self.default_path = os.path.join(self.cfg['project_path'], 'data', self.cfg['video_sets'][0])
        self.motifs_path = os.path.join(self.cfg['project_path'], 'results', self.cfg['video_sets'][0], 'VAME')
        self.original_data = None
        self.labeled_data = None
        self.unique_motifs = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Motif Visualizer')
        self.setGeometry(100, 100, 500, 250)
        
        layout = QVBoxLayout()

        # Original Data
        self.btn_load_orig = QPushButton('Select Original Data (.npy)')
        self.btn_load_orig.clicked.connect(self.load_original)
        layout.addWidget(self.btn_load_orig)
        self.lbl_orig = QLabel('No file selected')
        layout.addWidget(self.lbl_orig)

        # Reconstructed Data
        self.btn_load_recon = QPushButton('Select Labeled Data (.npy)')
        self.btn_load_recon.clicked.connect(self.load_labeled)
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

        # Feature Selector (2)  ✅ NEW
        h_layout2 = QHBoxLayout()
        h_layout2.addWidget(QLabel('Additional Feature :'))
        self.combo_feature2 = QComboBox()
        self.combo_feature2.addItems(self.columns_names)
        # optionally default to something else than feature1
        if len(self.columns_names) > 1:
            self.combo_feature2.setCurrentIndex(1)
        h_layout2.addWidget(self.combo_feature2)
        layout.addLayout(h_layout2)

        # Motif Selector
        h_layout_motifs = QHBoxLayout()
        h_layout_motifs.addWidget(QLabel('Motif:'))
        self.combo_motifs = QComboBox()
        h_layout_motifs.addWidget(self.combo_motifs)
        layout.addLayout(h_layout_motifs)

        # Plot single motif button
        self.btn_plot = QPushButton('Plot Selected Motif')
        self.btn_plot.clicked.connect(self.plot_data)
        layout.addWidget(self.btn_plot)

        #plot all motifs button
        self.btn_plot_all = QPushButton('Plot All Motifs')
        self.btn_plot_all.clicked.connect(self.plot_all_motifs)
        layout.addWidget(self.btn_plot_all)

        # Plot two features button ✅ NEW
        self.btn_plot_two = QPushButton('Plot Two Features + Motifs')
        self.btn_plot_two.clicked.connect(self.plot_two_features)
        layout.addWidget(self.btn_plot_two)


        self.setLayout(layout)

    def load_original(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open Original Data', self.default_path, 'Numpy Files (*.npy)')
        if fname:
            try:
                self.original_data = np.load(fname)
                self.lbl_orig.setText(os.path.basename(fname))
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def load_labeled(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open Labeled Data', self.motifs_path, 'Numpy Files (*.npy)')
        if fname:
            try:
                self.labeled_data = np.load(fname)
                self.labeled_data = self.labeled_data.flatten()
                self.lbl_recon.setText(os.path.basename(fname))
                
                # Get unique motifs and populate the combo box
                unique_motifs = np.unique(self.labeled_data)
                unique_motifs = unique_motifs[~np.isnan(unique_motifs)] # remove nan
                self.unique_motifs = unique_motifs[unique_motifs != -1]
                
                self.combo_motifs.clear()
                self.combo_motifs.addItems([str(int(m)) for m in unique_motifs])

            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def plot_data(self):
        if self.original_data is None or self.labeled_data is None:
            QMessageBox.warning(self, "Warning", "Please load both original and labeled data files.")
            return

        idx = self.combo_feature.currentIndex()
        
        try:
            orig = self.original_data
            labels = self.labeled_data
            
            # Select feature and flatten
            y_orig = orig[idx,:].flatten()

            # Ensure same length
            min_len = min(len(y_orig), len(labels))
            y_orig = y_orig[:min_len]
            labels = labels[:min_len]

            plt.figure(figsize=(20, 5))
            plt.plot(y_orig, label='Original', color ='k', alpha=0.5, linewidth=1)
            
            selected_motif_str = self.combo_motifs.currentText()
            if selected_motif_str:
                selected_motif = int(selected_motif_str)
                motif_times = np.where(labels == selected_motif)[0]
                if len(motif_times) > 0:
                    # Plot motif occurrences as vertical spans
                    starts = motif_times[np.where(np.diff(motif_times) != 1)[0] + 1]
                    starts = np.insert(starts, 0, motif_times[0])
                    ends = motif_times[np.where(np.diff(motif_times) != 1)[0]]
                    ends = np.append(ends, motif_times[-1])

                    for start, end in zip(starts, ends):
                        plt.axvspan(start, end + 1, color='red', alpha=0.3)

            plt.title(f'Motif "{self.combo_motifs.currentText()}" ')
            plt.xlabel('Time steps')
            plt.ylabel('Value')
            plt.legend()
            plt.tight_layout()
            plt.show()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error plotting data: {e}")
    
    '''
    plot all motifs found on the same graph
    '''
    def plot_all_motifs(self):
        if self.original_data is None or self.labeled_data is None:
            QMessageBox.warning(self, "Warning", "Please load both original and labeled data files.")
            return
        
        if self.unique_motifs is None or len(self.unique_motifs) == 0:
            QMessageBox.warning(self, "Warning", "No motifs found in labeled data.")
            return
        idx = self.combo_feature.currentIndex()

        try:
            orig = self.original_data
            labels = self.labeled_data
            
            # Select feature and flatten
            y_orig = orig[idx,:].flatten()

            # Ensure same length
            min_len = min(len(y_orig), len(labels))
            y_orig = y_orig[:min_len]
            labels = labels[:min_len]

            plt.figure(figsize=(20, 5))
            plt.plot(y_orig, label='Original', color ='k', alpha=0.5, linewidth=1)
            
            # colormap for motifs
            cmap = plt.cm.get_cmap('tab20', len(self.unique_motifs))

            for i, motif in enumerate(self.unique_motifs):
                motif_times = np.where(labels == motif)[0]
                if len(motif_times) == 0:
                    continue
                starts, ends = self._compute_segments(motif_times)
                for start, end in zip(starts, ends):
                    plt.axvspan(start, end + 1, color=cmap(i), alpha=0.3, label=f'Motif {motif}' if start == starts[0] else None)

            plt.title(f'All motifs on feature "{self.combo_feature.currentText()}"')
            plt.xlabel('Frames')
            plt.ylabel('Value')
            plt.legend()
            plt.tight_layout()
            plt.show()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error plotting all motifs: {e}")    
    
    '''
    plot two features with the motifs found
    '''
    def plot_two_features(self):
        if self.original_data is None or self.labeled_data is None:
            QMessageBox.warning(self, "Warning", "Please load both original and labeled data files.")
            return

        if self.unique_motifs is None or len(self.unique_motifs) == 0:
            QMessageBox.warning(self, "Warning", "No motifs found in labeled data.")
            return

        idx1 = self.combo_feature.currentIndex()
        idx2 = self.combo_feature2.currentIndex()

        try:
            orig = self.original_data
            labels = self.labeled_data

            y1 = orig[idx1, :].flatten()
            y2 = orig[idx2, :].flatten()

            min_len = min(len(y1), len(y2), len(labels))
            y1 = y1[:min_len]
            y2 = y2[:min_len]
            labels = labels[:min_len]

            fig, axes = plt.subplots(2, 1, figsize=(20, 7), sharex=True)

            axes[0].plot(y1, color='k', alpha=0.6, linewidth=1, label=self.combo_feature.currentText())
            axes[1].plot(y2, color='b', alpha=0.6, linewidth=1, label=self.combo_feature2.currentText())

            cmap = plt.cm.get_cmap('tab20', len(self.unique_motifs))

            # overlay all motifs on BOTH axes
            for i, motif in enumerate(self.unique_motifs):
                motif_times = np.where(labels == motif)[0]
                if len(motif_times) == 0:
                    continue
                starts, ends = self._compute_segments(motif_times)
                for j, (start, end) in enumerate(zip(starts, ends)):
                    for ax in axes:
                        ax.axvspan(
                            start, end + 1,
                            color=cmap(i), alpha=0.25,
                            label=f'Motif {int(motif)}' if (ax is axes[0] and j == 0) else None
                        )

            axes[0].set_ylabel('Value')
            axes[1].set_ylabel('Value')
            axes[1].set_xlabel('Frames')

            axes[0].legend(loc='upper right')
            # second legend: only feature name (to avoid duplicates)
            axes[1].legend(loc='upper right')

            fig.suptitle('Two Features + Motif Overlay', y=0.98)
            plt.tight_layout()
            plt.show()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error plotting two features: {e}")

    '''
    help function to compute start and end of each motif segment
    ''' 
    def _compute_segments(self, motif_times):
        """Helper to compute contiguous segments from time indices."""
        if len(motif_times) == 0:
            return [], []
        # find boundaries where the sequence breaks
        breaks = np.where(np.diff(motif_times) != 1)[0]
        starts = motif_times[breaks + 1] if len(breaks) > 0 else np.array([], dtype=int)
        starts = np.insert(starts, 0, motif_times[0])
        ends = motif_times[breaks] if len(breaks) > 0 else np.array([], dtype=int)
        ends = np.append(ends, motif_times[-1])
        return starts, ends
       

class MotifMarkerApp:
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
            self.gui = MotifMarkGUI(self.cfg, columns_names)
            self.gui.show()
            app.exec_()
        else:
            self.gui = MotifMarkGUI(self.cfg, columns_names)
            self.gui.show()

    def extract_landmarks_names(self):
        fname = self.filename[0] if isinstance(self.filename, list) else self.filename
        header = pd.read_csv(os.path.join(self.path_to_file,"videos","pose_estimation",fname+'.csv'), nrows=0)
        columns_names = header.columns.tolist()[1:]  # Skip the first column which is usually an index or frame number
        #remove the columns which its name contains the name score
        columns_names = [name for name in columns_names if 'score' not in name.lower()]
        return columns_names