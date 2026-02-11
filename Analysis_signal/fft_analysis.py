import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt


class fft_analysis:
   def __init__(self, data_path, labels_path, landmarks_information_csv, feature_name , label_choice):
        self.data_path = data_path
        self.labels_path = labels_path
        self.landmarks_information_csv = landmarks_information_csv
        self.feature_name = feature_name
        self.label_choice = label_choice


   def __call__(self, *args, **kwds):
        idx = self.find_feature_index()
        data = np.load(self.data_path)  # shape (num_frames, num_features)
        data = data.T
        labels = np.load(self.labels_path)  # shape (num_frames,)
        feature_data = data[0:len(labels), idx]  # Extract the specific feature column
       
        # Select data corresponding to the specified label
        # self.selected_data = feature_data[labels == self.label_choice]
        # self.selected_data = self.selected_data[0:400] 
        # # Perform FFT
        # self.fft_analysis()
        self.selected_data = feature_data
        #filter the a and band pass filter
        f_low = 2    # Hz
        f_high = 30  # Hz
        order = 4

        fs = 60  # Sampling frequency (frames per second)
       
        # Butterworth band-pass here
        b, a = butter(order, f_low, btype="high", fs=fs)
        self.selected_data = filtfilt(b, a, self.selected_data)

        #Perform spectrogram analysis of all the signal without selecting a specific label
        self.plot_spectrogram()
    


   def find_feature_index(self):
        header = pd.read_csv(self.landmarks_information_csv, nrows=0)
        columns_names = header.columns.tolist()[1:]  # Skip the first column which is usually an index or frame number
        columns_names = [name for name in columns_names if 'score' not in name.lower()]
        #remove the columns which its name contains the name score
        idx = columns_names.index(self.feature_name)
        return idx

   def fft_analysis(self):
        fs = 60  # Sampling frequency (frames per second)
        #remove dc component
        self.selected_data = self.selected_data - np.mean(self.selected_data)

        n = len(self.selected_data)  # Number of samples
        # Perform FFT
        amplitude = np.fft.fft(self.selected_data)
        freq = np.fft.fftfreq(n, d=1/fs)  # Frequency values
        

        mag = np.abs(amplitude)

       # ignore DC (0 Hz)
        mag[0] = 0

        idx_max = np.argmax(mag)
        dominant_freq = freq[idx_max]

        print(f"Dominant frequency: {dominant_freq:.2f} Hz")

        # ---- Plot ----
        fig, axs = plt.subplots(2, 1, figsize=(12, 8))
        axs[0].plot(self.selected_data)
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('Amplitude')
        axs[1].plot(freq, np.abs(amplitude))
        axs[1].set_xlabel('Frequency (Hz)')
        axs[1].set_ylabel('Amplitude')
        plt.tight_layout()
        plt.show()

    
   def plot_spectrogram(self):
        fs = 60  # Sampling frequency (frames per second)

        x = np.asarray(self.selected_data).squeeze()
       # x = x - np.mean(x)          # remove DC
        #x = np.nan_to_num(x)        # if any NaNs

        # Choose window length (in samples). With fs=60:
        
        nperseg = min(30, len(x))
        noverlap = int(0.75 * nperseg)

        n = len(x)
        t_signal = np.arange(n) / fs   # time axis (seconds)

        f, t, Sxx = signal.spectrogram(
            x,
            fs=fs,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            scaling="density",
            mode="psd"
        )

        # --- Plot ---
        fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        
        # Top: time-domain signal
        axs[0].plot(t_signal, x, color="black", linewidth=1)
        axs[0].grid(True)
        axs[0].set_xlabel("Time (s)")
        axs[0].set_ylabel("Signal")
        axs[0].set_title("Signal (time domain)")

        # Bottom: Spectrogram
        pcm = axs[1].pcolormesh( t,f,10 * np.log10(Sxx + 1e-12),shading="auto", vmin=-80, vmax=-30)
       # axs[1].set_ylim(0, 10)  # 0..30 Hz

        #axs[1].pcolormesh(t, f, 10 * np.log10(Sxx + 1e-12), shading="auto",vmin=-80, vmax=-30)
        axs[1].set_ylim(2, 20)  # 0..30 Hz
        axs[1].set_xlabel("Time (s)")
        axs[1].set_ylabel("Frequency (Hz)")
        axs[1].set_title("Spectrogram (PSD, dB)")
        #fig.colorbar(pcm, ax=axs[1], label="Power (dB)")
        plt.tight_layout()
        plt.show()

   


