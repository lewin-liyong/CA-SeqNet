import mne
import numpy as np
import numpy as np
import matplotlib.pyplot as plt




cleaned_epochs = mne.read_epochs('F:\Subjects_class\whl_blink\cleaned_epochs.fif')
cleaned_epochs.load_data()
cleaned_evokeds = cleaned_epochs.average()
cleaned_evokeds.plot(show=True, gfp=True)
plt.show(block=True)






signal_time = (0, 0.2)  # signal interval
noise_time = (-0.2, 0)    # noise interval

# Extract time index
times = cleaned_evokeds.times
signal_idx = np.where((times >= signal_time[0]) & (times <= signal_time[1]))[0]
noise_idx = np.where((times >= noise_time[0]) & (times <= noise_time[1]))[0]

# Calculate RMS of signal interval and noise interval
signal_rms = np.sqrt(np.mean(cleaned_evokeds.data[:, signal_idx] ** 2, axis=1))
noise_rms = np.sqrt(np.mean(cleaned_evokeds.data[:, noise_idx] ** 2, axis=1))

# Calculate SNR (in dB)
snr_db = 20 * np.log10(signal_rms / noise_rms)
# Output SNR of each channel
print("SNR (dB):", snr_db)

mean_snr = np.mean(snr_db)
# Output the average SNR of all channels
print(f"Mean SNR (dB): {mean_snr}")


















