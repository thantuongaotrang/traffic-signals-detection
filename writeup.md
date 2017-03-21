#**Traffic Sign Recognition** 
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/samples.png "Exapmle of Images"
[image2]: ./examples/data_summary.png "Visualization"
[image3]: ./examples/mean_std.png "Mean and std of the images"
[image4]: ./exapmles_from_internet/1.jpg "Traffic Sign 1"
[image5]: ./exapmles_from_internet/2.jpg "Traffic Sign 2"
[image6]: ./exapmles_from_internet/3.jpg "Traffic Sign 3"
[image7]: ./exapmles_from_internet/4.jpg "Traffic Sign 4"
[image8]: ./exapmles_from_internet/5.jpg "Traffic Sign 5"
[image9]: ./exapmles_from_internet/6.jpg "Traffic Sign 9"
[image10]: ./examples/img_pred.png "Prediction of new images"
[image11]: ./examples/img_pred_prob.png "Probabilties of new images"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

1. Here is a link to my [project code](https://github.com/yosoufe/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb) and [Writeup](https://github.com/yosoufe/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup.md)

### Dataset Exploration

1. Data Set Summary & Exploration

The code for this step is contained in the 2nd code cell of the IPython notebook.  

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the 3rd and 4th code cell of the IPython notebook. 

First some ramdom examples of images are plotted.

![alt text][image1]

then a graph, showing the number of repeatition of each class in training dataset is generated.

![alt text][image2]

### Design and Test a Model Architecture

1. Preprocessing

The code for this step is contained in the 5th code cell of the IPython notebook.

First I decided to convert the images to grayscale but without it, I have a better results. Maybe because using RGB data would use three times more number of parameters on first layer than a grayscale input.

I am normalizing the image using this equation: (input_image - mean)/std. The following image is showing the std and mean values like an image. The mean and std values are saved to be used later for other data to be feed into the network. Normalization helps have a faster convergance.

![alt text][image3]

3. Model Architecture
The code for my final model is located in the 6th cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x10 	|
| RELU					|												|
| DROPUOT					|	only for training											|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 14x14x20	|
| RELU					|												|
| DROPUOT					|	only for training											|
| Max pooling	      	| 2x2 stride,  outputs 5x5x20 				|
| Fully connected		| inputs 500, outputs 200									|
| RELU					|												|
| DROPUOT					|	only for training											|
| Fully connected		| inputs 200, outputs 100									|
| RELU					|												|
| DROPUOT					|	only for training											|
| Fully connected		| inputs 100, outputs 43									|
 


4. Model Training

The code for training the model is located in the 7-10th cells of the ipython notebook. 

* Optimizer: AdamOptimizer.

   This optimizer is decreasing the learning rate among each epoch, logarithmic, that causes faster convergence and less nosiy accuracy on test dataset.

* learning rate = 0.0005
* Regularization coefficient = 0.005
* EPOCHS = 100
* BATCH_SIZE = 500

5. Solution Approach

The code for calculating the accuracy of the model is located in the 9th cell of the Ipython notebook the function `evaluate`. The calculation is done in 10th and 14th cell.

My final model results were:
* training set accuracy of 99.2%
* validation set accuracy of 94.6% 
* test set accuracy of 92.9

If a well known architecture was chosen:
* What architecture was chosen and why?

   LeNet Architecture,mainly because recommended by Udacity. Of course convolution layers are usefull for image calssification tasks. Because it can find features of images itself and then using those features to classify the image. And feature detection is independent of the position in image because the Convolution layer is applied in all region of the image. Secondly it can be computed in parallel and it is fast enoough for GPUs.
   
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

   The accuracy percentages are all above 92 percent. It should be improved of course but I guess it is acceptable at this level.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8] ![alt text][image9]

Image 4 was not in the training dataset so results in wrong labeling. One of the images had some text label and interestingly, the model label it correctly.

2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the 14th cell of the Ipython notebook.

Here are the results of the prediction:

![alt text][image10]


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. The one that is wrong was out of the dataset.

3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook. Interestingly, the image which was not in the dataset, image 4, has the lowest maximum probabilty.

![alt text][image11]
