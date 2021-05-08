[//]: # (Image References)

[image1]: ./images/sample_dog_output.png "Sample Output"
[image2]: ./images/vgg16_model.png "VGG-16 Model Layers"
[image3]: ./images/vgg16_model_draw.png "VGG16 Model Figure"


## Project Overview
___
The goal of this project is to explore the machine learning approach to classifier images, in this case, dog's breed. Based on a picture of a dog, an algorithm will give an estimate of the breed of the dog. If the image of a person is given, the algorithm should reproduce the most similar dog breed. 

See the project [proposal](https://github.com/paulovsm/dog-breed-classifier/blob/master/proposal.pdfhttps://www.google.com) for more information.

![Sample Output][image1]

The algorithm should load an image as input and classify it as follows:

* In case of a dog, predict the dog's breed;
* In case of a human face, predict a dog's breed which that human face resembled it;
* In case of neither dog nor human, it should provide an error

In order to demonstrate the algorithm working a web application was developed. This web application was developed with Flask and made available online.

## Steps to achieve project goal:
___
* Import Datasets (Human faces and Dogs)
* Detect a human
* Detect a dog
* Create a CNN to Classify Dog Breeds
* Create a CNN to Classify Dog Breeds (using Transfer Learning)
* Develop a custom Algorithm
* Test algorithm

## Tools and Environments used on this project
___
* **Jupyter notebook / AWS Sagemaker Notebook**. It was used from data exploration to final algorithm tests
* **Jupyter notebook / Google Colab**. It was used from experimenting and testing models training leveraging the available GPU resources.
* **PyTorch**. It was used to develop the CNN models to predict the dog's breed
* **OpenCV**. It was used to identify the human face
* **Matplotlib**. It was used to show images and plot graphs
* **NumPy**, It was used to support large, multi-dimensional arrays and matrices, functions to operate on these arrays.

## Devices used for testing
___
* AWS Sagemaker notebook
	* Small instance for data exploration (ml.t2.medium) 
	* Large instance with GPU to speed up the training (ml.p2.8xlarge)
* Google Colab notebook with GPU support enable

To manually run through the code, you may simply follow this URL to open the notebook on [Google Colab](https://colab.research.google.com/github/paulovsm/dog-breed-classifier/blob/master/dog_app.ipynb).

## Demo Application
___
This web application was developed with Flask. In this simple application, I provide the trained model for the prediction. The user can simply upload an image and it will be forwarded to the result page after the inference/prediction process.

On the `app` folder the code of a small web project is available. The live demo was deployed on AWS EC2 and eventually will be accessible at this link: 

## Project Improvements
___
* Improved face detecting algorithm, replacing OpenCV Haar Cascades with some robust Object Detection network trained to detect faces or use Dlib capabilities to do so.
* Better control the scenario where both dog and human are present in the same picture
* Try different alternatives for classifier layers of the transfer learning model. Try to change only the FC layer and compare the performance.
* Expand training with more epochs and test the influence of the batch_size in the overall performance
* Try to improve model's accuracy  by fine-tuning hyperparameters and  testing different backbone models (Inception, ResNet, VGG19)
* Leverage SageMaker SDK and API for model training and model deployment purposes.