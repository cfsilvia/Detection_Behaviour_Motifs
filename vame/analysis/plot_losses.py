import os 
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from vame.util.auxiliary import read_config

class plot_losses:
    def __init__(self,config):
        #read config file
       config_file = Path(config).resolve()
       cfg = read_config(config_file)
       self.train_losses_file = os.path.join(cfg['project_path'],"model", "model_losses", "train_losses_VAME.npy")
       self.test_losses_file = os.path.join(cfg['project_path'],"model", "model_losses", "test_losses_VAME.npy")
       self.output_file = os.path.join(cfg['project_path'],"model", "losses.pdf")

    def __call__(self):
        self.plot_losses()



#####################################################
    def plot_losses(self):
        data_train = np.load(self.train_losses_file)
        data_test = np.load(self.test_losses_file)

        x_train = np.arange(len(data_train))
        x_test = np.arange(len(data_test))

        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 8), sharex=True)
        
        axes[0].plot(x_train, data_train, color='blue', linewidth=2)
        axes[0].set_ylabel("Train loss")

        axes[1].plot(x_test, data_test, color='red', linewidth=2)
        axes[1].set_ylabel("Test loss")
        axes[1].set_xlabel("epochs")
        
        plt.tight_layout()
        fig.savefig(self.output_file, format = "pdf", bbox_inches = "tight")
        plt.show()


