import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['STSong']
plt.rcParams['axes.unicode_minus']=False
import mne
from mne.preprocessing import ICA
import torch
import numpy as np
from model_predict import load_model, predict


epochs = mne.read_epochs('F:/raw_epoch_long/liyong/MEG/filter_2_48/auditory/epochs.fif')
model_path = '/RAID5/projects/fuxingwen/ly/class/outputs/1728769193.1415195/best_ckpt.pth'


# Fit source components
epochs_ICA = epochs.copy()
ICA_filt = ICA(n_components=12)
ICA_filt.fit(epochs)
sources = ICA_filt.get_sources(epochs)


# Identification and elimination of artifacts by epoch
clean_epochs = []
# Traverse each epoch
for i in range(len(epochs_ICA)):
    # Extract the current epoch
    temp_epochs = epochs_ICA.copy()
    temp_epochs1 = temp_epochs.drop([j for j in range(len(epochs_ICA)) if j != i]).copy()
    # Source component extraction
    ica_sources = sources._data[i]  # It is a NumPy array of [n_components, length]
    # Data transformation to adapt to prediction network
    input_data = ica_sources.T  # Transposed to [length, n_components] adaptive prediction network
    input_data = input_data[np.newaxis, :, :]  # Adding the batch dimension
    input_data = input_data.transpose(2,0,1)  # Adjust to [n_components,batch_size,length]
    input_data = torch.Tensor(input_data)  # Convert to Tensor
    if torch.cuda.is_available():
        input_data = input_data.cuda()  # If you use a GPU

    # Predicted source component type        (channels,batch_size,sequence_length)
    model = load_model(model_path)
    predicted_labels = predict(model, input_data)
    print("Predicted Labels:", predicted_labels)
    predicted_labels.tolist()
    indices = [i for i, value in enumerate(predicted_labels) if value in (0, 1)]
    print("Predicted ICAs:", indices)
    # Artifact removal
    ICA_filt.exclude = indices
    ICA_filt.apply(temp_epochs1, exclude=ICA_filt.exclude)
    clean_epochs.append(temp_epochs1.get_data())
cleaned_epochs_data = np.concatenate(clean_epochs, axis=0)
# Epochs format recovery
cleaned_epochs = mne.EpochsArray(cleaned_epochs_data, epochs.info, tmin=-0.5,baseline=(-0.2, 0))
cleaned_epochs.set_montage(epochs.get_montage())
cleaned_epochs.save('F:/raw_epoch_long/liyong/MEG/filter_2_48/auditory/cleaned_epochs.fif', overwrite=True)