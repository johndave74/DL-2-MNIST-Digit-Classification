# DL-2-MNIST-Digit-Classification
This project demonstrates how to build and train a deep learning model using TensorFlow and Keras to classify clothing items in the Fashion MNIST dataset. The goal is to accurately predict the category of fashion items based on grayscale images (28x28 pixels).


## 📂 Dataset

The [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset contains 70,000 grayscale images of 10 fashion categories such as T-shirts, trousers, bags, and ankle boots:

* 60,000 images for training
* 10,000 images for testing

Each image is 28x28 pixels in size, and labels range from 0 to 9.

## 🧰 Libraries Used

| Library             | Purpose                                                   |
| ------------------- | --------------------------------------------------------- |
| `tensorflow.keras`  | Deep learning model creation, training, and evaluation    |
| `matplotlib.pyplot` | Visualization of training progress and sample predictions |
| `numpy`             | Numerical operations and array manipulations              |

## 📊 Workflow Summary

### 1. **Data Loading & Preprocessing**

* Load the Fashion MNIST dataset from `keras.datasets`.
* Normalize the pixel values by dividing by 255.0 to scale them between 0 and 1.
* Visualize sample images with class labels to understand the dataset.

### 2. **Model Architecture**

A sequential CNN model was built using the following layers:

* `Conv2D` with 32 filters
* `MaxPooling2D`
* `Conv2D` with 64 filters
* `MaxPooling2D`
* `Flatten`
* `Dense` layer with 128 neurons
* Output `Dense` layer with 10 units and softmax activation

### 3. **Model Compilation & Training**

* **Optimizer**: Adam
* **Loss function**: Sparse Categorical Crossentropy
* **Metrics**: Accuracy
* Trained for 10 epochs with validation data.

### 4. **Model Evaluation**

* Evaluated model performance on the test set.
* Final accuracy: \~90% (exact figure from output).
* Plotted training and validation accuracy/loss curves.

### 5. **Predictions & Visualization**

* Generated predictions on test images.
* Visualized some predictions alongside the true labels to interpret model performance.

## 🔍 Insights

* CNNs are well-suited for image classification problems and significantly outperform dense networks on image datasets.
* Overfitting was controlled by a relatively shallow network and using validation data.
* The model generalized well, with similar accuracy on training and test datasets.

## 📈 Results

* **Training Accuracy**: \~91%
* **Test Accuracy**: \~89–90%
* The model was able to accurately classify most of the fashion items in the test set.

## 🧪 Future Improvements

* Implement data augmentation to further generalize the model.
* Use more advanced architectures like ResNet or MobileNet for better performance.
* Try transfer learning with pre-trained weights on similar tasks.

## 🚀 How to Run

1. Clone the repo:

   ```bash
   git clone https://github.com/your-username/fashion-mnist-classification.git
   cd fashion-mnist-classification
   ```

2. Install dependencies:

   ```bash
   pip install tensorflow matplotlib numpy
   ```

3. Run the notebook:
   Open `DL_3_Fashion_MNIST_Image_Classification.ipynb` in Jupyter Notebook or VS Code and run all cells.
