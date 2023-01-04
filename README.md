Link to the data:

https://drive.google.com/drive/folders/1YNrQe3_xqwBHLtZ12dVR4bHbG8vvLqyb?usp=sharing

Link to data compatible with lazy loading:

https://drive.google.com/drive/folders/1OkittwtywCKIZWSaQ5MPd3rp4lMqe2W2?usp=sharing

# Project Report
# 4-fingered robot hand Prediction

## Method
### DataSet Class
In order to effectively and efficiently operate on the data for my machine learning project, I created a custom Image Dataset class. This class provides several useful functions for loading, processing, and analyzing data.

The ‘init’ function allows for the selection of either train data or test data, which is then loaded from the lazydata folder. This allows for flexibility and customization in data selection and preparation.

The ‘len’ function simply returns the length of the data that has been loaded, providing an easy way to determine the size of the dataset.

The ‘getitem’ function is responsible for a number of important tasks. It reads the .png image data (image0, image1, and image2), as well as the depth data. It then calculates the mean and standard deviation for all four of these data types across 3396 data points. Using these values, it creates normalizers for each of the image data types and uses transforms.Normalize to normalize them. The depth data is then divided by 1000 and normalized using the formula (X-mean)/std. Finally, the ground truth data (Y) is read and multiplied by 1000. The final format returned by getitem is a tuple containing the processed images, depth data, and ground truth, in the format (image0, image1, image2, depth), Y.

### Data Processing
In order to optimize the data for use in my machine learning model, I customized a series of transformation operations that are applied to the image data that is read. After using ToPILImage(), I mainly used the Grayscale() and ColorJitter() methods. This is because the images contain a lot of unnecessary noise, and their colors can affect the precision of the model. Since the main focus of the model is the black palm and fingers with white fingertips, I used grayscale to highlight this feature and increase the contrast coefficient in order to make it more obvious and easier for the model to learn efficiently. The contrast parameter for ColorJitter is set to (.7, .8), while the other parameters such as brightness, saturation, and hue are all set to 0, which means they are unchanged. Finally, I used ToTensor() to convert the image back to the Tensor format. Overall, this customized series of transformation operations allows for effective and efficient data processing and preparation for use in the machine learning model.

### Data Loading
Based on the preparation steps that I completed previously, I then loaded the training data into an instance of the customized dataset class. I selected a batch size of 16 and set the shuffle option to True. I then created a data loader for the training data so that it could be used in the subsequent steps of my model training process. To verify that the correct number of data points had been loaded, I printed the length of the training data.

In addition to creating the training data, I also created a local validation dataset, which was randomly split from the original dataset and was not used during the training process. This allowed me to evaluate the performance of my model on unseen data, giving me an unbiased estimate of its skill. However, during the final stages of model training, I chose to merge the validation data back into the training data in order to boost the performance of the model by increasing the amount of data it had available for training.

### Customized Train Method
I wrote my customized train method. The loss function I use is MSELoss.

### Customized Test Method
I also created a customized Test method. The loss function is consistent with the one used in the Train method - MSELoss. The accuracy is calculated per epoch, by averaging all the mse loss value generated.

### Customized Resnet50 Model Setup
I use the resnet50 model from torchvision.models library. The fully-connected layer fc and first convolutional layer conv1 are customized in order to fit the data (12 channels) input specifically for this problem.

### Start Training Process
I choose to use Adam as my optimizer. The learning rate is set to 0.01 and the momentum is set to default.
I then ran the train method from epoch=0 to epoch=69 recursively to train my model.

### Model State Saving
I used torch.save() method to save the current state of my model trained (including parameters)

### Submission File Generation
This part of the code generates the submission file. The id list is loaded from the original testX.pt file since I didn’t read it from lazy data in my customized dataset. The final output is divided by 1000 since during the training I multiplied it by 1000 and the model result would be 1000 times greater than the original one.

## Experimental Results
### Epoch Value
![Epoch Value](/imgs/Epoch.png)

### Preprocessing Method Selection
![Preprocessing Method Selection](/imgs/Preprocessing.png)

### Optimizer Selection
![Optimizer](/imgs/Optimizer.png)

## Discussion
### Epoch Value
Based on the data, it appears that the loss and validation error both decrease as the epoch increases. This suggests that the model's performance improves as it trains for more epochs. The validation error reaches its lowest point at epoch 40, after which it begins to increase slightly. This could indicate that the model is beginning to overfit to the training data. However, the decrease trend continues again, implying that the point epoch=40 may be an outlier. Overall, the results show that increasing the number of epochs can improve the model's performance, but it may be necessary to carefully balance the number of epochs to prevent overfitting.

### Preprocessing Method Selection
Notice that, with no change on my current preprocessing methods:
Based on the data, it appears that adding the 'CenterCrop(200)' and 'RandomHorizontalFlip()' transformations to the model can affect its performance. The 'CenterCrop(200)' and 'Resize()' transformations can make it difficult for the model to learn the connections between the depths at each pixel and the new resized images, resulting in a lower prediction error. On the other hand, the 'RandomHorizontalFlip()' transformation does not improve the accuracy of the model because the camera positions are assumed to be fixed, and there is no need to flip the images horizontally. In fact, overfitting on the training data with respect to the horizontal position can actually improve the model's ability to predict on unseen data.

### Optimizer Selection
Based on the data, it appears that the optimizer has a significant influence on the error at epoch 10. The Adam and SGD optimizers perform the best, with error values of 242.1 and 390.7, respectively. The Adagrad optimizer has a much higher error value of 517.9, and the RMSprop optimizer has an even higher error value of 670.3. This suggests that the Adam and SGD optimizers are more effective at reducing the error during this training. It is worth noting that the error values for all of the optimizers are relatively high, indicating that there is room for further improvement in the model's performance. Additionally, the learning rate and momentum values may also have an impact on the model's performance, but this is not clear from the given data.

## Future Work
### Epoch Value
I could use a larger epoch size for training the data. So far the largest one that I used is 100. I wonder whether there will be significant improvement if I increase it to 300 or 500. But that requires a much better GPU… I’ve already spent $100 on Colab for this project 🙁

### Preprocessing Method Selection
I could try to find more various image preprocessing techniques. I actually have an idea but don’t know how to implement it. I tried to crop the three images differently so that the useless borders are discarded and only the hand is kept. However, there is a difficulty: if I crop them differently and then resize them to be (224,224) again, I haven’t come up with an idea of how to crop and resize the depth. I could work more on this.

### Optimizer Selection
I could also try using different optimizers and learning rate values to optimize my model. And for the four that I analyzed in the Experimental Results section, I could try larger epoch size and test their performance.

### Model Selection
Instead of using the ResNet50 model, I could try other pre-built ResNet models as well as Inception and DenseNet, and could also build a CNN model by myself. 
