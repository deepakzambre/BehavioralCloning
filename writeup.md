# Behavioral Cloning
---
## File organization
My project includes the following files:
* _model.py_: Used to generate models, it can be invoked as
`python model.py`. It loads data from `./data/` directory, fits models, checkpoints models to `./model_checkpoints/` and creates a plot of training and validation loss.

  _Note: it saves all models to checkpoint directory and does not save the final model to `./model.h5`; we manually select a model from the checkpoint directory based on loss graph._
* _drive.py_: for driving the car in autonomous mode, can be invoked as `python drive.py model.h5`
* _model.h5_: pre-trained solution model for this project
* _video.mp4_: video of vehicle driving autonomously around track_1 for one lap using model.h5

## Model Architecture and Training Strategy
I have used NVIDIA's self driving car neural network architecture in my project. It can be understood by referring to following diagram:

![](./model.png?raw=true)

I have used Adam Optimizer to tune parameters for the model.
To prevent overfitting, I selected model for epoch after which validation loss increases or does not change much while training loss decreases. To allow this model selection, model.py creates a plot of validation loss and training loss vs number of epochs (plot is shown below, selected solution model is from previous run of model.py). I ran model.py multiple times to get a good model which drives car autonomously without leaving the track.

![](./model_loss.png?raw=true)

## Training data collection

I augmented sample_data that is provided with project with me driving car in center lane for one lap `./data/lap`. I observed that model trained using this data was drifting the car on left side. Hence I augmented data by flipping the images and steering angle when loading the data in-memory (refer method `load_data`).

Thereafter, depending on models performance in autonomous mode, I augmented data with specific maneuvers as shown below:

Edge to center 1

![](.\data\specific_training_1\IMG\center_2017_08_05_00_35_35_619.jpg?raw=true)
![](.\data\specific_training_1\IMG\center_2017_08_05_00_35_37_502.jpg?raw=true)
![](.\data\specific_training_1\IMG\center_2017_08_05_00_35_39_422.jpg?raw=true)

Edge to center 2

![](.\data\specific_training_1\IMG\center_2017_08_05_00_36_22_508.jpg?raw=true)
![](.\data\specific_training_1\IMG\center_2017_08_05_00_36_26_149.jpg?raw=true)
![](.\data\specific_training_1\IMG\center_2017_08_05_00_36_28_592.jpg?raw=true)

Edge to center on bridge

![](.\data\specific_training_1\IMG\center_2017_08_05_00_37_56_959.jpg?raw=true)
![](.\data\specific_training_1\IMG\center_2017_08_05_00_38_01_337.jpg?raw=true)
![](.\data\specific_training_1\IMG\center_2017_08_05_00_38_03_569.jpg?raw=true)

Soil to center after bridge

![](.\data\specific_training_1\IMG\center_2017_08_05_00_40_06_742.jpg?raw=true)
![](.\data\specific_training_1\IMG\center_2017_08_05_00_40_07_998.jpg?raw=true)
![](.\data\specific_training_1\IMG\center_2017_08_05_00_40_10_652.jpg?raw=true)

Similarly, I collected sample data for other situations like entering bridge, exiting bridge, etc. All of specific maneuvers data is located in `./data/specific_training_1`, `./data/specific_training_2` and `./data/specific_training_3` folders.

## Video of car being driven autonomously

Solution video is located at `./video.mp4`.
