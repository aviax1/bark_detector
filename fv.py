import pandas as pd
import numpy as np
pd.plotting.register_matplotlib_converters()
import librosa,csv
#https://www.kaggle.com/prabhavsingh/urbansound8k-classification/

size = 8732
df = pd.read_csv("data/UrbanSound8K.csv")
with open('data/feature_vector.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["file", "label", "feature_vector"])
    for i in range(size):
        file_name = 'data/fold' + str(df["fold"][i]) + '/' + df["slice_file_name"][i]
        X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mels = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
        rw = [df["slice_file_name"][i], df["classID"][i]]
        for i2 in range(len(mels)):
            rw.append(mels[i2])
        writer.writerow(rw)
        print("row ",i+1,"/",size)
