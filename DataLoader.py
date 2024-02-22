import os 
import pandas as pd
import pyedflib
from tqdm import tqdm
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import keras

# ---------------------------------------------------------------------
#  Utility functions
# ---------------------------------------------------------------------

def split_into_epochs(x, epoch_size=32000, axis=1):
  """   Split the data at specified axis. The resulting array is obtained by concatenating
        at axis 0 the splits obtained
  
  Args:
      x: the numpy array to split into epochs 
      epoch_size: the length of the splits
      axis: the axis on which the x array is split
  """

  if x.shape[axis] <= epoch_size:
      return x

  epoch_size = int(epoch_size)
  split_indices = range(epoch_size, x.shape[axis], epoch_size)

  return np.concatenate(np.array_split(x, split_indices, axis=axis)[:-1], axis = 0)


# ---------------------------------------------------------------------
#  DataLoader class
# ---------------------------------------------------------------------

class DataLoader():
    """ Load data from local files, enable to get train and test data from it
        Creates raw_records and raw_respevents by reading all the subjects' records
        Creates clean_records and clean_respevents from raw_records and raw_respevents
        Here, clean_records simply removes features that have missing values (i.e. is empty for at least one subject) 
    """

    def __init__(self, folder_path, epoch_duration = 5*60):

        assert os.path.exists(folder_path)

        self.db_folder = folder_path 

        ## Read data
        self.subject_details = pd.read_excel(os.path.join(self.db_folder, "SubjectDetails.xls"))
        _, signal_headers, _ = pyedflib.highlevel.read_edf(os.path.join(self.db_folder,"ucddb002.rec"))
        self.signal_headers_table = pd.DataFrame(signal_headers)

        ## Compute numpy arrays from the data
        # we store the raw signals for each record in this array
        self._load_raw_data()
        # Creates the actual data to use for training/testing from the raw data
        self._create_clean_data()

    def _load_raw_data(self):

        self.raw_records = np.zeros((self.n_records, self.signal_len, self.n_signals), dtype="float32")
        self.raw_respevents = np.zeros((self.n_records, self.signal_len, 1), dtype="bool")

        for entry in tqdm(os.scandir(self.db_folder)):
          rootname, ext = os.path.splitext(entry.name)
          study_number = rootname[:8].upper()
          if not study_number.startswith("UCDDB"):
            continue
          subject_index, = np.where((self.subject_details["Study Number"] == study_number))[0]
          if ext == ".rec":
            signals, signal_headers, header = pyedflib.highlevel.read_edf(entry.path)
            for sig, sig_hdr in zip(signals, signal_headers):
              try:
                signal_index, = np.where((self.signal_headers_table["label"] == sig_hdr["label"]))[0]
              except ValueError:
                if sig_hdr["label"] == "Soud":
                  signal_index = 7
              if sig_hdr["sample_rate"] != 128:
                q = int(self.sample_rate//sig_hdr["sample_rate"])
                sig = np.repeat(sig, q)
                sig = sig[:self.signal_len]
                self.raw_records[subject_index,:,signal_index] = sig.astype("float32")
          elif rootname.endswith("respevt"):
            respevents = pd.read_fwf(os.path.join(self.db_folder,rootname + ".txt"), widths=[10,10,8,9,8,8,6,7,7,5], skiprows=[0,1,2], skipfooter=1, engine='python', names=["Time", "Type", "PB/CS", "Duration", "Low", "%Drop", "Snore", "Arousal", "Rate", "Change"])
            respevents["Time"] = (pd.to_datetime(respevents["Time"]) - pd.to_datetime(self.subject_details.loc[subject_index, "PSG Start Time"])).astype("timedelta64[s]")%(3600*24)
            respevents["Time"] = pd.to_timedelta(respevents["Time"], unit="s")
            for _, event in respevents.iterrows():
              onset = int(self.sample_rate*event["Time"].total_seconds())
              offset = onset + int(self.sample_rate*event["Duration"])
              self.raw_respevents[subject_index, onset:offset] = 1

    def _create_clean_data(self):
        # We remove the features that have missing values (here, the features that are removed
        # are actually missing for most of the subjects, so we deem it worthy removing them)
        mask_features_to_remove = np.any(~np.any(self.raw_records, axis = 1), axis=0)
        
        self.clean_records = np.delete(self.raw_records, mask_features_to_remove, axis=2) # self.raw_records#
        self.clean_respevents = self.raw_respevents


    def train_test_split(self, test_size=0.3, epoch_duration = 5*60):

        ## Split data into training and testing sets

        X_train, X_test, y_train, y_test = train_test_split(self.clean_records, self.clean_respevents, test_size=test_size)

        ## Split the testing and training sets into epochs of duration epoch_duration

        epoch_size = self.sample_rate * epoch_duration 
        # epoch size should be a multiple of 32 to be used by Unet_1d
        epoch_size -= epoch_size % 32 

        X_train = split_into_epochs(X_train, epoch_size=epoch_size)
        X_test = split_into_epochs(X_test, epoch_size=epoch_size)
        y_train = split_into_epochs(y_train, epoch_size=epoch_size)
        y_test = split_into_epochs(y_test, epoch_size=epoch_size)

        return X_train, X_test, y_train, y_test

    def plot_data(self, subject_id, raw = False):

      records = self.raw_records if raw else self.clean_records
      respevents = self.raw_respevents if raw else self.clean_respevents

      # The last line is the label binary signal, the others are input signals for the model
      onset = int(datetime.timedelta(hours=1, minutes=29, seconds=30).total_seconds()*self.sample_rate)
      offset = int(onset+120*self.sample_rate)
      
      fig, axes = plt.subplots(nrows = self.n_signals+1, sharex=True, figsize=(30,12))
      for n, (ax, label) in enumerate(zip(axes[:-1], self.signal_headers_table["label"].to_list())):
        ax.plot(records[subject_id, onset:offset,n])
        ax.set_ylabel(label)
      fig.suptitle('Subject id: {}'.format(subject_id))
      axes[-1].plot(respevents[subject_id, onset:offset], "r")  

        

    @property
    def n_records(self):
        return len(self.subject_details)

    @property
    def duration(self):
        return self.subject_details['Study Duration (hr)'].min()*3600 // 4

    @property
    def sample_rate(self):
        return self.signal_headers_table['sample_rate'].max()

    @property
    def signal_len(self):
        return int(self.sample_rate*self.duration)

    @property
    def n_signals(self):
       return len(self.signal_headers_table)
