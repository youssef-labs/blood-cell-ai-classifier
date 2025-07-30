# AI Blood Cell Classifier

A convolutional neural network (CNN) built with TensorFlow and Keras to classify images of four different types of white blood cells (leukocytes). I completed this project in a 7-day sprint to create a functional deep learning model for a real-world biomedical problem.

## About The Project

The goal of this project is to accurately classify blood cell images into their respective categories: Eosinophil, Lymphocyte, Monocyte, and Neutrophil. The project demonstrates an end-to-end machine learning workflow, encompassing data processing, model training, and evaluation.

### Technologies Used

* Python
* TensorFlow & Keras
* Scikit-learn
* NumPy & Pandas
* Matplotlib & Seaborn
* Jupyter Notebook

## Usage

To run this project locally:

1.  Clone the repository:
    `git clone https://github.com/youssef-labs/blood-cell-ai-classifier.git`
2.  Create the conda environment and install the required packages.
3.  Launch Jupyter Notebook and open the `blood_cell_classifier.ipynb` file.

## Results

After training for 15 epochs on a dataset of nearly 10,000 images, the model achieved the following performance on the unseen test set:

* **Final Test Accuracy:** 67.9%

### Training History

The graph below shows the model's learning progress over the 15 epochs. The validation accuracy closely tracks the training accuracy, indicating that the model is learning effectively without significant overfitting.

![Training and Validation Curves](https://github.com/youssef-labs/blood-cell-ai-classifier/blob/main/results-graph.png)

## Future Improvements

Given more time, the next steps to improve model performance would be:
* **Data Augmentation:** Artificially increase the size and variance of the training dataset by applying random rotations, flips, and zooms to the images.
* **Transfer Learning:** Implement a more complex, pre-trained model architecture (such as VGG16 or ResNet50) to leverage knowledge from larger datasets.
