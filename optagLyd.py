import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt

Fs=8000
duration =10

# sd.default.device = (0, 1)
    
recording = sd.rec(
        int(duration * Fs),
        samplerate=Fs,
        channels=1,
        dtype='int16'
    )

sd.wait()  # Blocking until recording is finished

print("End of Recording.")
    
x = recording.flatten() # signal stored in x


# Write your code here

