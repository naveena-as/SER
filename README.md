# SER
Machine Learning model for Speech Emotion Recognition (SER) in Python

### Install
Here the data is not in textual format, but in the form of audio files. Hence we are required to use Python libraries for analysing audios, [Librosa](https://librosa.org/doc/latest/index.html), [Soundfile](https://pypi.org/project/SoundFile/), and [PyAudio](https://pypi.org/project/PyAudio/)

This project was done using Python in Google Colaboratory. Hence not all packages were required to be installed explicitly. However, other IDEs (like JupyterLab) can be used as well. This might require additional package installations.
You may install these with the help of the following command:
```bash
pip install librosa soundfile numpy sklearn pyaudio
```

### The dataset
The RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset sums up to 24.8 GB. For this project, we use a reduced version of the same, available [here](https://drive.google.com/file/d/1wWsrN2Ep7x6lWqOXfr4rpKGYrJhWc8z7/view) for download. This consists of 24 folders, each for an actor (12 male and 12 female), with 60 audio files per folder. The file naming convention gives us details about the type of data it contains. Read [this](https://zenodo.org/record/1188976#.YLYRzybhVpk) for further information.

SER is ususally done with the help of Neural Network models, since they give better predictions and high accuracy. Here, our objective is to build the model using basic ML classifiers, though we have tried fitting the [MLP classifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html) as well. There are 8 different emotions in the dataset and classification into 8 subcategories will give very low accuracy, we have considered only 4 major emotions - neutral, angry, sad, happy - for the purpose of classification.

### Implementation
This project gives insight to two different approaches in building the model
1. Use Python libraries to extract features and then perform train and test to make predictions directly
2. Extract various features from the files, tabulate them, then select those features which gives an appreciable prediction accuracy

### Approach 1
After extracting the features, the train test split was performed for two different ratios - 75:25 and 90:10. To justify that the latter split doesn't cause overfitting of the model, we introduce four test samples (one per emotion) from another dataset - [EmoDB dataset](https://www.kaggle.com/piyushagni5/berlin-database-of-emotional-speech-emodb) - and prove the same.

### Approach 2
Here we extract the features, compute their mean values and create a CSV file of the same. Each audio file is be represented by a row in the CSV file, which also contains the corresponding actor number and emotion extracted from the filename. With this CSV file as data, we can easily fit a model.

### Model fitting and accuracy analysis
The model has been fitted with various classifiers like KNN, SVM, Decision Tree, etc. and the accuracy obtained has been analysed as bar charts. (Refer to `AccuracyAnalysis_90:10.jpeg` and `AccuracyAnalysis_75:25.png` in the list of files)
