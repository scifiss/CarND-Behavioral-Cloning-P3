
# Behavioral Cloning Project

**Rebecca Gao**
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/angleDistForDifferentDataset.png "Angle distribution from different datasets"
[image2]: ./examples/angleDistNoZeros.png "Final angle distribution"
[image3]: ./examples/structure.png "Architecture"
[image4]: ./examples/center_2017_08_04_17_12_02_042.jpg "center driving"
[image5]: ./examples/center_2017_09_18_14_09_18_254.jpg "Recovery Image 1"
[image6]: ./examples/center_2017_09_18_14_09_24_027.jpg "Recovery Image 2"
[image7]: ./examples/center_2017_09_18_14_09_27_036.jpg "Recovery Image 3"
[image8]: ./examples/center_2017_09_15_15_18_08_086.jpg "track 2"
[image9]: ./examples/augImages.png "orig noisy flipped"
[image10]: ./examples/angleDist.png "orig angle distribution"
[image11]: ./examples/center_2017_09_20_15_06_13_588smooth.jpg "smooth driving on the bridge"
[image12]: ./examples/center_2017_09_21_13_32_43_676recover.jpg "recovery driving on the bridge"
[image13]: ./examples/hit_the_bridge.png "hit the bridge"
[image14]: ./examples/TrainValLoss.png "loss"

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
The car is running a little wavy.

5th trial:
The top 80 and bottom 25 pixels by width are cropped off the image.
Results: similar with last trial

6th trial: 
Decreasing correction factor from 0.2 to 0.1.
Results: the car is running stabler.

7th trial:
Increase training data from 0.8 of the total to 0.9 of the total.	
Results: worse. the car falls off the lane.

8th trial:
try NVIDIA network (https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/)

'''

model = Sequential()
model.add(Lambda(lambda x: x/255.0-0.5,
        input_shape=( height, width,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))

model.add(Convolution2D(24, 5, 5,subsample=(2,2), activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(36, 5, 5,subsample=(2,2), activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(48, 5, 5,subsample=(2,2), activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

'''

Results: worse. the car runs very slow and falls off the lane.

9th trial:
go back to Lenet network. Add max pooling layers and dropout layers.
Results: more stable, but always hit the left side of the black bridge and crash.

![alt text][image13]

10th trial:
After scutinizing the images, I find my data along the bridge all have positive angles, accordingly the dataset along the bridge is very positively biased. Therefore, I drive the car two more times along the bridge, one is smooth driving, the other is recovering several times from both sides. I also add Gaussian noises and flipped the whole images to augment data sample.

![alt text][image11]
![alt text][image12]

After all dataset is to be fed into the model, it fails due to running out of memory. I then use the training/validation generator to implement the model.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 103-126) consisted of a convolution neural network with the following layers and layer sizes.

Here is a visualization of the architecture 

![alt text][image3]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded four laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image4]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to go back to the center once it slides to the sides. These images show what a recovery looks like starting from the left side to the center:

![alt text][image5]

![alt text][image6]

![alt text][image7]

I also drive and record twice along the brick bridge, since I noticed (during training) the car is easily to drive off to the side or to the out space after the bridge.


Then I repeated this process on track two in order to get more data points.

![alt text][image8]

To augment the dataset, I add Gaussian noise (mean=0, var=0.001, so the variation locally is around 3) to all original dataset, using their same angle measurements. I also flipped images and angles to double the original dataset. For example, here is an image that has been added noised and flipped:

![alt text][image9]

After the collection process, I had 16966 number of data points (each point has center, left, and right images). Below is the steering angle distribution.

![alt text][image10]

As is visible, there are too many zeros that'll biased the dataset with straight driving. To train my model, only non-zero angles are added. Here is the angle distribution of final dataset.
From each dataset (non-zero angle): 

![alt text][image1]

The entire dataset (non-zero angle):

![alt text][image2]

I then preprocessed this data by 
* crop the image (cut off the upper 75 pixels in height and the lower 25 pixels in height)
* normalize the values of each color channel(mean=0, var=1)

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 9 as evidenced by the validation loss starts to increase from epoch 10. (from the plot, the learning rate 0.0003 is still too high)

![alt text][image14]

I used an adam optimizer, although manually training the learning rate wasn't necessary, I still tune the learning rate to be lower and both training and validation loss decrease a lot.


Further Discussion
For me, the most uncomfortable part part is preprocessing, which is subject to subjective judgement. 
(1) we would like to get rid of uninteresting part of a image.
For how much upper part to be cropped off the image for training, it is done by hand selection. Different riding environment may have different noisy part. For example, for track 1, the upper 70-80 pixels can be cut off, but for track 2, the upper 80-85 pixels should be cut off. Also, how far ahead in the lane should be referenced in driving also depends on speed. The higher the speed, the farther the lane ahead should be taken into consideration.
(2) we don't have a lot of samples of extreme situations in real life. But in order to recover from the sides, we need to create a lot of artificial samples of recovering from lane edges to the center. This method of augmentation is not generalizable.
(3) Currently my car can't run on track 2, which means my model is not generalized. 



