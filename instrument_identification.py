#!/usr/bin/env python
# coding: utf-8

# # Automated Instrument Identification
# #### Nicholas Gibb

import numpy as np
import pandas as pd
import librosa
from sklearn.ensemble import RandomForestClassifier

# Load training and test data to dataframe
df_train = pd.read_json("data/nsynth-train/examples.json", orient="index")
df_test = pd.read_json("data/nsynth-test/examples.json", orient="index")

# Group the df_train by instrument family, and then sample 4500 from each
samples_per_instrument = 4500
df_train_reduced = df_train.groupby(
    "instrument_family", as_index=False, group_keys=False
).apply(lambda df: df.sample(samples_per_instrument))

# Remove synth lead (instrument 9) as it is missing from test data
df_train_reduced = df_train_reduced[df_train_reduced["instrument_family"] != 9]

# This function takes a file and returns features in an array
def extract_audio_features(file):

    # Load the wav file
    y, sr = librosa.load(file)

    # Determine if sound is harmonic or percussive
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    if np.mean(y_harmonic) > np.mean(y_percussive):
        harmonic = 1
    else:
        harmonic = 0

    # Mel-frequency cepstral coefficients
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    # Temporal averaging
    mfcc = np.mean(mfcc, axis=1)

    # Mel-scaled spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    # Temporal averaging
    mel_spectrogram = np.mean(mel_spectrogram, axis=1)

    # Determine chroma energy
    chroma = librosa.feature.chroma_cens(y=y, sr=sr)
    # Temporal averaging
    chroma = np.mean(chroma, axis=1)

    # Determine spectral contrast
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    # Temporal averaging
    spectral_contrast = np.mean(spectral_contrast, axis=1)

    # Return audio features as array
    audio_features = [harmonic, mfcc, mel_spectrogram, chroma, spectral_contrast]
    return audio_features


instrument_types = [
    "bass",
    "brass",
    "flute",
    "guitar",
    "keyboard",
    "mallet",
    "organ",
    "reed",
    "string",
    "synth_lead",
    "vocal",
]

# Function to extract instrument from filename
def instrument_code(filename):
    for instrument in instrument_types:
        if instrument in filename:
            instrument_index = instrument_types.index(instrument)
            return instrument_index
    else:  # Should never get this far
        return None


# #### Feature extraction from test audio data

# Create dictionary to store all test features
# This will be converted to dataframe after
test_features_dict = {}
filenames_test = df_test.index.tolist()

# loop over every audio file in test set
for file in filenames_test:

    # feature extraction; function returns array
    file_path = "data/nsynth-test/audio/" + file + ".wav"
    audio_features = extract_audio_features(file_path)

    # store features in dictionary for each file
    test_features_dict[file] = audio_features

# Convert dict to dataframe
features_test = pd.DataFrame.from_dict(
    test_features_dict,
    orient="index",
    columns=["harmonic", "mfcc", "spectro", "chroma", "contrast"],
)

# Extract mfccs
mfcc_test = pd.DataFrame(features_test.mfcc.values.tolist(), index=features_test.index)
mfcc_test = mfcc_test.add_prefix("mfcc_")

# Extract spectro
spectro_test = pd.DataFrame(
    features_test.spectro.values.tolist(), index=features_test.index
)
spectro_test = spectro_test.add_prefix("spectro_")

# Extract chroma
chroma_test = pd.DataFrame(
    features_test.chroma.values.tolist(), index=features_test.index
)
chroma_test = chroma_test.add_prefix("chroma_")

# Extract contrast
contrast_test = pd.DataFrame(
    features_test.contrast.values.tolist(), index=features_test.index
)
contrast_test = chroma_test.add_prefix("contrast_")

# Drop the old columns
features_test = features_test.drop(
    labels=["mfcc", "spectro", "chroma", "contrast"], axis=1
)

# concatenate
df_features_test = pd.concat(
    [features_test, mfcc_test, spectro_test, chroma_test, contrast_test],
    axis=1,
    join="inner",
)

# df_features_test doesn't have the target labels... add it now
# Extract the instrument from the file name
targets_test = []
for name in df_features_test.index.tolist():
    targets_test.append(instrument_code(name))
# Add new column to dataframe -- the instrument name
df_features_test["targets"] = targets_test


# Now we do it all again, this time with the training data

# WARNING: Executing this cell will take several hours!

# Dictionary to store all training features
# This will be converted to dataframe after
training_features_dict = {}

filenames_train = df_train_reduced.index.tolist()

# loop over every audio file in training set
for file in filenames_train:

    # feature extraction; function returns array
    file_path = "data/nsynth-train/audio/" + file + ".wav"
    features = extract_audio_features(file_path)

    # store features in dictionary for each file
    training_features_dict[file] = features

# Convert dict to dataframe
features_train = pd.DataFrame.from_dict(
    training_features_dict,
    orient="index",
    columns=["harmonic", "mfcc", "spectro", "chroma", "contrast"],
)


# extract mfccs from array stored in one column to seperate columns
mfcc_train = pd.DataFrame(
    features_train.mfcc.values.tolist(), index=features_train.index
)
# convenient column renaming function in pandas
# generates mfcc_0, mfcc_1, ..
mfcc_train = mfcc_train.add_prefix("mfcc_")

# Extract spectro
spectro_train = pd.DataFrame(
    features_train.spectro.values.tolist(), index=features_train.index
)
spectro_train = spectro_train.add_prefix("spectro_")


# Extract chroma
chroma_train = pd.DataFrame(
    features_train.chroma.values.tolist(), index=features_train.index
)
chroma_train = chroma_train.add_prefix("chroma_")


# Extract contrast
contrast_train = pd.DataFrame(
    features_train.contrast.values.tolist(), index=features_train.index
)
contrast_train = chroma_train.add_prefix("contrast_")

# Drop the old columns
features_train = features_train.drop(
    labels=["mfcc", "spectro", "chroma", "contrast"], axis=1
)

# Concatenate all the columns
df_features_train = pd.concat(
    [features_train, mfcc_train, spectro_train, chroma_train, contrast_train],
    axis=1,
    join="inner",
)

# df_features_train doesn't have the target labels... add it now
# Extract the instrument from the file name
targets_train = []
for name in df_features_train.index.tolist():
    targets_train.append(instrument_code(name))

# Add new column to dataframe -- the instrument name
df_features_train["targets"] = targets_train


# ### Random forest model

# get training and testing data
X_train = df_features_train.drop(labels=["targets"], axis=1)
y_train = df_features_train["targets"]

X_test = df_features_test.drop(labels=["targets"], axis=1)
y_test = df_features_test["targets"]

# Generate the random forest
clf_Rf = RandomForestClassifier(n_estimators=20, max_depth=50, warm_start=True)

clf_Rf.fit(X_train, y_train)

y_pred_RF = clf_Rf.predict(X_test)

random_forest_accuracy = np.mean(y_pred_RF == y_test)
print("Random forest algorithm accuracy: {0:.1%}".format(random_forest_accuracy))


# ### Credits

# Nadim Kawwa, *[Instrument classification on the NSynth dataset using supervised learning and CNNs](https://github.com/NadimKawwa/NSynth)*. 2019.
