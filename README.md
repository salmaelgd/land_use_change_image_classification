# land_use_change_image_classification
The rapid expansion of urban areas and the increasing demand for agricultural land have led to signif-
icant changes in land use patterns worldwide. Monitoring and understanding these changes are crucial
for effective land management, environmental conservation, and sustainable development. Remote sens-
ing technologies coupled with machine learning algorithms offer a powerful approach for analyzing and
classifying land use changes over time.
In this project, the focus is on the indispensable role of machine learning (ML), specifically Convo-
lutional Neural Networks (CNNs), in accurately categorizing land use changes across a diverse dataset.
The utilization of CNN models enables the analysis of satellite imagery to identify and classify variations
in land cover over time. The overarching objective is to automate the classification process, providing
invaluable insights into urbanization, agricultural expansion, and deforestation trends within each land
class
# Data Collection
I utilized the EuroSAT dataset, which is available through TensorFlow, to access a comprehensive col-
lection of satellite images. This dataset comprises a total of 27,000 images, derived from the Sentinel-2
satellite, and covers a broad spectrum of land cover types across 13 spectral bands. These include 10 dis-
tinct classes: AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop,
Residential, River, and SeaLake. For the purpose of this study, I specifically focused on utilizing the
”rgb”, R (red), G (green), and B (blue), version of the dataset. The dataset was divided into 80% for
training and 20% for testing. 
# Data Preprocessing

To enhance the diversity and robustness of the training dataset, I applied several image augmentation techniques. These techniques are essential for improving the model's generalization ability and making it more resilient to variations in real-world data. The following augmentation techniques were applied:

- **Rescaling:** Each pixel value in the images was rescaled to a range between 0 and 1 by dividing by 255.
- **Rotation Range:** Random rotation of the images was performed, with a rotation range of up to 20 degrees. This allows the model to learn features from different orientations of the objects.
- **Horizontal Flip:** Random horizontal flipping of the images was applied to help the model learn features that are invariant to left-right orientation.
- **Vertical Flip:** Random vertical flipping of the images was performed, introducing additional variations in the training data.
- **Zoom Range:** Random zooming was applied to the images, with a zoom range of 0.2. This enables the model to recognize objects at different scales.

# Convolutional Neural Network Model

## Components of Convolutional Neural Network Architecture

The Convolutional Neural Network (CNN) architecture used in this model consists of multiple layers designed to extract features from input images and classify them into one of the ten land use classes. This section provides concise explanations of the activation functions, optimizer, loss function, max pooling, and dropout techniques utilized in the model.

### Activation Functions

- **ReLU:** The ReLU (Rectified Linear Unit) activation function introduces non-linearity by outputting the input directly if it is positive, otherwise outputting zero. This function helps in addressing the vanishing gradient problem and accelerating the convergence of gradient-based optimization algorithms.
- **Softmax:** The softmax function transforms a vector of K real values into a vector of K real values that sum to 1, allowing them to be interpreted as probabilities.

### Optimizer

- **Adam:** The Adam optimizer is an adaptive learning rate optimization algorithm used for training deep neural networks. It dynamically adjusts the learning rate for each parameter based on the magnitude of recent gradients and the exponentially decaying average of past gradients. This adaptive mechanism allows Adam to converge faster and more reliably compared to traditional gradient descent methods.

### Loss Function

- **Categorical Cross-Entropy Loss:** This loss function is used when the number of classes is greater than two, applying categorical cross-entropy to optimize model performance.

## Architecture

The architecture begins with an input layer of size 128x128x3, representing the dimensions of the input images with three color channels (RGB). The model then applies a series of convolutional layers, each followed by batch normalization and max pooling layers to extract and downsample features while maintaining spatial information. Additionally, dropout layers with a dropout rate of 0.3 are incorporated after each max pooling layer to reduce overfitting by randomly dropping 30% of the neurons. The convolutional layers use ReLU activation functions. After the final convolutional layer, the feature maps are flattened into a 1-dimensional array and passed through two fully connected dense layers with ReLU activation functions, and 500 and 32 units, respectively, each followed by batch normalization and dropout layers. Finally, the output layer consists of 10 units with softmax activation.

## Metrics and Training

In the model evaluation process, a custom F1 score metric function was introduced in addition to the standard accuracy, precision, and recall metrics provided by Keras. The custom F1 score function calculates the harmonic mean of precision and recall, offering a comprehensive assessment of the model’s performance. The model was trained over 20 epochs with a batch size of 128.

# Results and Discussion

## Model Training Progress

- **Accuracy:** The model's accuracy improved from 65.89% to 94.21%.
- **Recall:** Recall rose from 52.90% to 93.24%.
- **Precision:** Precision reached 95.09%.
- **F1-Score:** The F1-score reached 94.13%.

### Training Results

| Data         | Accuracy | Recall  | Precision | F1-Score |
|--------------|----------|---------|-----------|----------|
| Train Data   | 0.9421   | 0.9324  | 0.9509    | 0.9413   |

## Test Data Performance

The evaluation of the model on the test dataset reveals strong performance:

| Data         | Loss    | Accuracy | Recall  | Precision | F1-Score |
|--------------|---------|----------|---------|-----------|----------|
| Test Data    | 0.8642  | 0.7685   | 0.7585  | 0.7837    | 0.7713   |

The model demonstrates effective learning and predictive capabilities, achieving nearly 77% accuracy on the test dataset. The overall accuracy indicates its effectiveness in correctly classifying a significant portion of instances across all classes.

