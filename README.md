# Instrument Identification

A classification model for identifying instruments based on sounds in the NSynth dataset.

![musical-instruments-clipart](https://user-images.githubusercontent.com/3398606/59603636-eb350680-90d7-11e9-8f5e-c185f54cfd59.jpg)

## Installation and usage
This project requires Python 3.x and the following Python libraries:

- pandas
- numpy
- librosa
- scikit-learn

Furthermore you will also have to have software installed to run and execute a Jupyter Notebook. Instructions to do so can be found on the [Jupyter website](https://jupyter.org/install). Note that you cannot execute a Notebook on the Github website -- you must start your own Jupyter server.

This repository also contains a raw Python script (converted from the Notebook. However, it is highly reccomended to execute the script within the Notebook.

To use the script, simply execute the cells in the Notebook. An introduction, methodology, and dicussion of the project is also included within the Notebook as markdown text.

## Data 

The NSynth dataset is used in this project. This dataset is free and publically-available. However, it is too large to be included in this repository. 

It can be downloaded [here](https://magenta.tensorflow.org/datasets/nsynth). In order for the code to work properly, the data must be in the correct subfolder: 
```
.
├───README.md
├───Instrument_identification.ipynb
├───data
│   ├───nsynth-test
|   |   ├───audio
|   |   ├───examples.json
│   └───nsynth-train
|   |   ├───audio
|   |   ├───examples.json
```

## Credits
Thank you to Github user NadimKawwa who published a similar analysis of the NSynth data, in much greater depth: https://github.com/NadimKawwa/NSynth