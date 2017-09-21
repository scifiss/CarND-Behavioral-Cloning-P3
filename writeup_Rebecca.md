#**Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/angleDistForDifferentDataset.png "Angle distribution from different datasets"
[image2]: ./examples/angleDistNoZeros.png "Final angle distribution"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### _Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation._  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* video.mp4 can replay the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of the following layers: (model.py lines 101-121) 
 - [x] normalize the image RGB values using Keras lambda layer
 - [x] cropp off the upper and lower part of the images, resulted in 60x320 images
 - [x] 2 convolutional layers with 5x5 filter sizes and depths between 6 and 16, each followed by 'relu' activation function and max pooling layers. Now the units are of size 12x77, depth =16
 - [x] flat into one layer
 - [x] fully connected layer of 120, 84, 10, and 1 nodes. 

#### 2. Attempts to reduce overfitting in the model
The model contains 2 dropout layers in order to reduce overfitting (model.py lines 111,116). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 68-70). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, but I still tune the learning rate from 0.0001 to 0.001(model.py line 122). A default learning rate of 0.01 make the learning speed fall into plateau too early.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. Bad training data could mislead the system to learn. For example, when driving towards the left edge of the road, before the driver realize so as to turn right hard, the system records data with negative angles and actually it should have positive data. Also, the driver can turn right too hard,making the car continue to turn right when it is already in the center.

I used a combination of center lane driving, recovering from the left and right sides of the road, drive on the opposite direction, etc. For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

At first, I tried to collect raw data as much as possible.
(1) with track 1, driving in the center normally and in the opposite direction
(2） with track 1, recording recoveries from left and right sides of the lane to the center
(3) with track 2, driving normally on the right lane

For the first 4 trials, I run for 7 epochs. They all give monotonically decreasing training and validation errors, and validation errors happen to be lower than training errors. So no over fitting yet.

1st trial: using a very simple network with Lambda for normalization, one convolution2D and a Dense.
Result: the car rides all the way to the left, like a circle.

2nd trial: using a Lenet network: Lambda, two Convolution2D with MaxPooling2D followed, Flatten, and 3 Dense (fully connected).
Result: the car runs all the way to the right, like a circle.

The first 2 trials results in very large steering angles: 30-45 degrees, and the training errors are in hundreds.

3rd trial: 
all images and measurements with angles=0 are removed. I did this because a) there are a lot of zero angles that will not give much information. b) The zeros are bad noises, since I was not a "perfect" driver, I can't immediately turn right or left when the car is too close to the left or right. The zero angles give the wrong information that in those situations before I respond to steer the car, the car is already not in the center.
Results: the car can run for 10+ seconds on the road.

4th trial:
(1) flipped images are added to the dataset
(2) Left and right images are added to the dataset, with a correction factor of 0.2, as suggested in the course.
	Left image is labeled with measured angle + 0.2
	Right image is labeled with measured angle - 0.2
Results: the car stays in the lane, until it runs off to the big open area that is connected with the lane (after the bridge).
The major improvements seem to be due to the increased dataset.

5th trial:
The top 80 and bottom 25 pixels by width are cropped off the image.
Results: similar with last trial

6th trial: 
Decreasing correction factor from 0.2 to 0.1.
Results: the car is running stabler.

7th trial:
Increase training data from 0.8 of the total to 0.9 of the total.	
Results: worse. the car falls off the lane.

7th trial:
try NVIDIA network (https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/)
Results: worse. the car runs very slow and falls off the lane.

8th trial:
go back to Lenet network. Add max pooling layers and dropout layers.
Results: more stable, but always hit the left side of the black bridge and crash.

After scutinizing the images, I find my data along the bridge all have positive angles. So the dataset along the bridge is very biased. Therefore, I drive the car two more times along the bridge, one is smooth driving, the other is recovering several times from both sides. I also add Gaussian noises and flipped the whole images to augment data sample.



Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...
* crop the image


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.


Further Discussion
For me, the most uncomfortable part part is preprocessing, which is subject to subjective judgement. 
(1) we would like to get rid of uninteresting part of a image.
For how much upper part to be cropped off the image for training, it is done by hand selection. Different riding environment may have different noisy part. For example, for track 1, the upper 70-80 pixels can be cut off, but for track 2, the upper 80-85 pixels should be cut off. Also, how far ahead in the lane should be referenced in driving also depends on speed. The higher the speed, the farther the lane ahead should be taken into consideration.
(2) we don't have a lot of samples of extreme data in real life. But in order to recover from the sides, we need to create a lot of artificial samples of recovering from lane edges to the center. This method of augmentation is not generalizable.




