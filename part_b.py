import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
from scipy.special import expit
import urllib
import time

ERRORS = []


class LogisticRegressionAlgorithm(object):
    def __init__(self, inputs, labels, learning_rate=.1, itertions=20000, test_amount=10000):
        # scaling input to fit best to the model
        self.inputs = self.prepare_input(inputs)  # flattened data with the beginning, size inputs (number of images, 784)
        self.labels = labels

        # split the data to train set and test set
        self.inputs_train, self.inputs_test, self.labels_train, self.labels_test = train_test_split(self.inputs, self.labels, test_size=test_amount)
        self.learning_rate = learning_rate
        self.iterations = itertions

    def prepare_input(self, inputs):
        """Scale input to range 0.01 - 1
        This is done because a zero input would kill the update of the weights
        because of the nature of sigmoid function"""
        scaled_inputs = np.true_divide(inputs, 255 * 0.99) + 0.01
        return scaled_inputs

    def plot_image(self, image_number, inputs):
        """Plots one image out of the images
        This function is just for debugging or visualizing, it is not used in calculations"""
        image = np.zeros((28, 28))
        for i in range(0, 28):
            for j in range(0, 28):
                pix = 28 * i + j
                image[i, j] = inputs[image_number, pix]
        plt.imshow(image, cmap='gray')
        plt.show()

    def initial_weights(self, number_of_inputs):
        """Initial weights, returns a vector of size 785x1 of random numbers between 0-1.
        we subtract 0.5 because the weights can be negative."""
        return np.random.rand(number_of_inputs, 1) - 0.5

    def sigmoid(self, x):
        return expit(x)

    def predict(self, weights, inputs):
        """Returns the sigmoid of weight.T * inputs
        Return is of shape (number of images, ) or more correct to say (number of classes, number of images)
        Each value in the array is the probability (between 0-1) of the label == 1 given the corresponding weights
        Parameters:
            weights: shape (784,1) or more correct is to say (784, number of classes)
            inputs: shape (784, number of images)"""
        number_of_inputs = inputs.shape[1]
        z = np.dot(weights.T, inputs)[0]
        z = np.reshape(z, (number_of_inputs, ))
        return self.sigmoid(z)

    def cross_entropy_error(self, labels, predictions):
        """
        The cross entropy error was developed in class
        Returns a scalar which represents the error

        inputs:(784, number of images)  each column is an image
        labels: (number of _images, )
        predictions: (number of images, )
        weights:(784,1)

        error = (labels*log(predictions) + (1-labels)*log(1-predictions) ) / -len(labels)
        """
        number_of_inputs = len(labels)

        # Take the error when label=1
        class1_error = labels * np.log(predictions)

        # Take the error when label=0
        class0_error = (1 - labels) * np.log(1 - predictions)

        # Take the sum of both costs
        error = np.sum(class1_error + class0_error)

        # Take the average cost
        error = error / number_of_inputs * -1

        return error

    def get_gradient(self, inputs, labels, predictions):
        """
        Returns the vector of the gradient, shape (784,1)
        inputs:(784, number of images)  each column is an image
        labels: (number of _images, )
        weights:(784,1)
        predictions: (number of images, )
        gradient = inputs(predictions - labels)"""
        number_of_inputs = len(labels)
        labels = np.reshape(labels, (number_of_inputs, 1))  # labels is now number_of_images x 1
        predictions = np.reshape(predictions, (number_of_inputs, 1))  # labels is now number_of_images x 1

        # Returns a (784,1) matrix holding 784 partial derivatives, one for each weight
        gradient = np.dot(inputs, predictions - labels)

        # Take the average error derivative for each weight
        gradient /= number_of_inputs

        return gradient  # shape (784,1)

    def update_weights(self, weights, gradient, learning_rate):
        """
        Update weights using gradient descent
        Returns weights of shape (784,1)
        Parameters:
            gradient: (number of images, )
            weights:(784,1)
            learning_rate: scalar between 0-1
        """

        # Multiply the gradient by the learning rate
        gradient *= learning_rate

        # Subtract from the weights to minimize error
        weights -= gradient

        return weights

    def decision_boundary_function(self, probability):
        """Helper function for classify"""
        return 1 if probability >= .5 else 0

    def classify(self, predictions):
        """
        The predictions are probabilities, now we need to classify them.
        We say that if the probability is greater (or equal) than 0.5 its class 1 and less than 0.5 is class 0
        input  - N element array of predictions between 0 and 1
            example: [0.5, 0.7, 0.2, 0.4, 0.6]
        output - N element array of 0s (False) and 1s (True)
            example corresponding to the input example [1, 1, 0, 0, 1]
        """
        decision_boundary = np.vectorize(self.decision_boundary_function)
        return decision_boundary(predictions).flatten()

    def classify_multiclass(self, predictions):
        """Turns the probabilities into classes, the class with the highest probability is the estimator
        Returns and array of shape (number of images, ) which each element holds the predicted label for the image"""
        return np.argmax(predictions, axis=0)


    def train_one_class(self, number_to_classify, inputs=None, labels=None, iterations=None, learning_rate=None):

        if inputs is None:
            inputs = self.inputs_train.T  # shape (784, 60000) where each column is an image
        if labels is None:
            labels = self.labels_train  # shape (60000, )
        if iterations is None:
            iterations=self.iterations
        if learning_rate is None:
            learning_rate = self.learning_rate

        # preparing the labels for binary classification with two classes (number to classify or not)
        # shape (60000, )
        labels = self.one_vs_all_labels(labels=labels, number_to_classify=number_to_classify)  # labels are now 1 or 0
        weights = self.initial_weights(inputs.shape[0])  # shape (784,1)
        error_history = []
        start_time = time.time()
        for i in range(iterations):
            # Get predictions of all images
            predictions = self.predict(weights=weights, inputs=inputs)  # shape (60000, )

            # Get the gradient
            gradient = self.get_gradient(inputs=inputs, labels=labels, predictions=predictions)  # shape (784, 1)

            # Update weights using gradient descent
            weights = self.update_weights(weights=weights, gradient=gradient, learning_rate=learning_rate)

            # Calculate error for plotting purposes
            error = self.cross_entropy_error(labels=labels, predictions=predictions)  # scalar

            error_history.append(error)

        elapsed_time = time.time() - start_time
        print("Finished training class {}\nThe training took {:0.3f} seconds\n".format(number_to_classify, elapsed_time))
        return weights, error_history

    def train_all_classes(self, iterations=None, learning_rate=None):
        """Trains all 10 classes as part of multi-class linear regression
        Returns an array of size 784x10 which each column is the optimal weights for a class.
        e.g returns np.array([w0, w1, ..., w9]) which each w is size 784x1"""

        print("Started training, this will take a few minutes...")
        if iterations is None:
            iterations = self.iterations
        if learning_rate is None:
            learning_rate = self.learning_rate

        start_time = time.time()

        weights = None
        for i in range(10):
            # Find the optimal weights of each class (weights i is size 784x1)
            weights_i = self.train_one_class(number_to_classify=i, learning_rate=learning_rate, iterations=iterations)[0]

            # save the optimal weights to form an array of optimal weights of each class
            if i == 0:
                weights = weights_i
            else:
                weights = np.hstack((weights, weights_i))

        elapsed_time = time.time() - start_time
        print("*********\nFinished Training all classes.\nThe training took {:0.3f} seconds".format(elapsed_time))
        return weights

    def test_one_class(self, weights, number_to_classify, inputs=None, labels=None):
        """Test the classifier and returns the true positive, true negative, false positive and false negative
            Parameters: inputs - shape (784, test_set_size) which each column has a different image
                        labels - shape (test set size which is 10000, ) which each element has the label of the image
                        weights - a vector of shape (784, 1)
                        number_to_classify - the class number being tested (0 to 9)"""

        if inputs is None:
            inputs = self.inputs_test.T  # shape (784, 10000) where each column is an image
        if labels is None:
            labels = self.labels_test  # shape (10000, )

        # get the labels ready
        labels = self.one_vs_all_labels(labels=labels, number_to_classify=number_to_classify)  # shape (10000, )

        # get predictions (probability row vector)
        predictions = self.predict(weights=weights, inputs=inputs)  # shape (10000, )

        # turn the probabilities into classes
        predicted_classes = self.classify(predictions=predictions)  # shape (10000, )

        true_pos = np.sum((predicted_classes == labels)[predicted_classes == 1])
        true_neg = np.sum((predicted_classes == labels)[predicted_classes == 0])
        false_pos = np.sum((predicted_classes != labels)[predicted_classes == 1])
        false_neg = np.sum((predicted_classes != labels)[predicted_classes == 0])

        return true_pos, true_neg, false_pos, false_neg

    def predict_multiclass_helper(self, weights, inputs):
        """Returns the sigmoid of weight.T * inputs
        Return value is of shape (number of classes, number of images)
        Each value in the array is the probability (between 0-1) of the label == 1 given the corresponding weights
        Parameters:
            weights: shape (784, number of classes)
            inputs: shape (784, number of images)"""

        z = np.dot(weights.T, inputs)
        return self.sigmoid(z)

    def predict_multiclass(self, weights, inputs):
        """Predicts the output of the multi-class linear regression
        Parameters: weights - shape (784, 10) which is column is the optimal weights of each class
                    inputs -  shape (784, number_of_images)
                    output is shape (number of classes(10), number of images) the some of each column is 1"""

        predictions = self.predict_multiclass_helper(weights=weights, inputs=inputs)

        # Normalizing each column to be between 0-1
        normalized_predictions = predictions / predictions.sum(axis=0, keepdims=True)
        return normalized_predictions  # shape (10, number of images)

    def test_multiclass(self, weights, inputs=None, labels=None):
        """Test the classifier and returns confusion matrix
        Parameters: inputs - size of 784 x 10000 which each column has a different image
        labels -  shape (10000, ) which each element has the label of the image
        weights - a vector of size 784 x 10 which is column has the optimal weights of each class"""

        if inputs is None:
            inputs = self.inputs_test.T  # inputs is now in shape (784, 10000) each column is an image
        if labels is None:
            labels = self.labels_test  # shape (10000, )

        predictions = self.predict_multiclass(weights=weights, inputs=inputs)
        classes = self.classify_multiclass(predictions=predictions)

        # initialize the confusion matrix to zeros
        # rows are predicted class and columns are actual class
        confusion_matrix = np.zeros((10,10), dtype="int32")

        test_set_size = inputs.shape[1]  # number of images in the test set
        for i in range(test_set_size):
            current_label = int(labels[i])
            current_predicted_class = classes[i]
            confusion_matrix[current_predicted_class][current_label] += 1

        return confusion_matrix

    def test_all_classes(self, weights, inputs=None, labels=None):
        """Test all the classes of the multi-class linear regression.
        This function is a helper for plotting the confusion tables of all the classes
        Parameters: weights - an array of size 785x10, each column has the optimal weights of each class
                    like the returned value of train_all_classes"""
        if inputs is None:
            inputs = self.inputs_test.T  # shape (784, 10000) where each column is an image
        if labels is None:
            labels = self.labels_test  # shape (10000, )

        # resulting_test... is in format [[true_pos0, true_neg0, false_pos0, false_neg0], [true_pos1, true_neg1, false_pos1, false_neg1],...]
        resulting_test_of_all_classes = []
        for i in range(10):
            weights_i = np.reshape(weights[:, i], (784, 1))
            true_pos, true_neg, false_pos, false_neg = self.test_one_class(weights=weights_i, number_to_classify=i,
                                                                           inputs=inputs, labels=labels)
            resulting_test_of_all_classes.append([true_pos, true_neg, false_pos, false_neg])

        return resulting_test_of_all_classes

    def one_vs_all_labels(self, labels, number_to_classify):
        """One vs all means using a binary classifier with more than two available outputs
        The label of the wanted class will be 1 and the rest will be 0"""
        modified_labels = np.where(labels == number_to_classify, 1, 0)  # modified labels is now 1 or 0 depending on the number to classify
        return modified_labels

    def calculate_accuracy(self, confusion_matrix):
        """Claculates the accuracy of the classifier"""
        return np.trace(confusion_matrix)/np.sum(confusion_matrix)

    def calculate_sensitivity(self, true_pos, false_neg):
        """Calculate the sensitivity of a classifier"""
        return  true_pos/(true_pos+false_neg)

    def plot_confusion_table(self, true_pos, true_neg, false_pos, false_neg, number_to_classify, show_plot=False):
        """Plots the confusion table of a specific class.
        Where the class is the number_to_classify"""

        sensitivity = self.calculate_sensitivity(true_pos, false_neg)
        table = np.array([[true_pos, false_pos], [false_neg, true_neg]])
        axis_names = ["{}".format(number_to_classify), "Not {}".format(number_to_classify)]

        plt.figure(number_to_classify)
        ax = plt.gca()
        ax.matshow(table, cmap=plt.get_cmap('Blues'))

        for (i, j), z in np.ndenumerate(table):
            if i == 0:
                if j == 0:
                    string = "True Positive"
                else:
                    string = "False Positive"
            else:
                if j == 0:
                    string = "False Negative"
                else:
                    string = "True Negative"

            ax.text(i, j, string + '\n{}'.format(z), ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

        tick_marks = np.arange(len(axis_names))
        plt.xticks(tick_marks, axis_names)
        plt.yticks(tick_marks, axis_names)
        plt.title("Confusion Table of number {}\n".format(number_to_classify))
        plt.ylabel('Actual Class')
        plt.xlabel('Predicted Class\nsensitivity={:0.4f}'.format(sensitivity))
        if show_plot:
            plt.show()

    def plot_confusion_table_of_all_classes(self, test_results, show_table=False):

        for i in range(10):
            true_pos, true_neg, false_pos, false_neg = test_results[i]
            self.plot_confusion_table(true_pos=true_pos, true_neg=true_neg, false_pos=false_pos,
                                      false_neg=false_neg, number_to_classify=i)
        if show_table:
            plt.show()


    def plot_confusion_matrix(self, confusion_matrix, show_plot=False):
        """Plots the confusion table of the multi-class PLA.
        Parameters: confusion_matrix - size 10x10
                    Rows are predicted class and columns are actual class as returned from test_multiclass"""

        accuracy = self.calculate_accuracy(confusion_matrix=confusion_matrix)

        axis_names = list(range(10))

        plt.figure(10)
        ax = plt.gca()
        ax.matshow(confusion_matrix, cmap=plt.get_cmap('Blues'))

        # Putting the value of each cell in its place and surrounding it with a box so you could see the text better
        for (i, j), z in np.ndenumerate(confusion_matrix):
            ax.text(i, j, '{}'.format(z), ha='center', va='center', fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

        # setting x and y axis to be the class numbers
        tick_marks = np.arange(len(axis_names))
        plt.xticks(tick_marks, axis_names)
        plt.yticks(tick_marks, axis_names)

        # setting title and labels for the axis
        plt.title("Confusion Matrix of the Multi-Class Linear Regression\n")
        plt.ylabel('Predicted Class')
        plt.xlabel('Actual Class\naccuracy={:0.4f} %'.format(accuracy*100))
        if show_plot:
            plt.show()


    def plot_confusion_matrix_and_tables(self, confusion_matrix, test_results):
        """Plots the confusion matrix and the confusion table of each class
        Parameters: confusion_matrix - size 10x10 as returned from test_multiclass
                    test_results - and array of tp, tn, fp, fn as returned from test_all_classes_part_a"""

        self.plot_confusion_table_of_all_classes(test_results)
        self.plot_confusion_matrix(confusion_matrix)
        plt.show()

    def plot_train_loss_of_all_classes(self, train_losses):
        """Plots the training loss.
        train_loss: array of arrays in this format: [[error history 0], [error history 1], ...].
        Each error history was logged every iteration, otherwise the x axis is wrong."""

        num_plots = len(train_losses)

        # Plot several different functions...
        labels = []
        for i in range(num_plots):
            plt.plot(train_losses[i])
            labels.append("  Class {}".format(i))

        plt.legend(labels, ncol=4, loc='upper center',
                   bbox_to_anchor=[0.5, 1.1],
                   columnspacing=1.0, labelspacing=0.0,
                   handletextpad=0.0, handlelength=1.5,
                   fancybox=True, shadow=True)
        plt.ylabel("Train Loss")
        plt.xlabel("Iteration")

        plt.show()

    def save_weights(self, weights, path=r"./multiclass_weights_part_b.npy"):
        """Save weights for further use"""
        np.save(path, weights)

    def get_weights_from_file(self, path=r"./multiclass_weights_part_b.npy"):
        """Load weights previously saved in a file"""
        return np.load(path)


def import_mnist():
    mnist_alternative_url = r"https://github.com/amplab/datascience-sp14/raw/master/lab7/mldata/mnist-original.mat"
    mnist_path = "./mnist-original.mat"
    response = urllib.request.urlopen(mnist_alternative_url)
    with open(mnist_path, "wb") as f:
        content = response.read()
        f.write(content)

    mnist_raw = loadmat(mnist_path)
    mnist = {
        "data": mnist_raw["data"].T,
        "target": mnist_raw["label"][0],
        "COL_NAMES": ["label", "data"],
        "DESCR": "mldata.org dataset: mnist-original",
    }
    print("Success in loading MNIST data set")
    return mnist


def main(save_weights=False, use_weights_from_file=False):
    # Get the mnist data set
    mnist = import_mnist()

    # instantiate a class of the Logistic Regression Algorithm with the mnist data set
    lra = LogisticRegressionAlgorithm(inputs=mnist["data"], labels=mnist["target"], learning_rate=1, itertions=300)

    # Train all the 10 classes of the multi-class linear regression
    # 10 classes, one for each digit of the mnist data set
    # Training is on the training set which is 60,000 images
    if use_weights_from_file:
        optimal_weights_of_all_classes = lra.get_weights_from_file()
    else:
        optimal_weights_of_all_classes = lra.train_all_classes()  # size 784x10 which each column is the optimal weights for a class.

    if save_weights:
        lra.save_weights(weights=optimal_weights_of_all_classes)

    # Testing each separate class
    # Testing is on the testing set which is 10,000 images
    test_results = lra.test_all_classes(optimal_weights_of_all_classes)  # See lra.test_all_classes_part_a for return value

    # Get the confusion matrix
    # Rows are predicted class and columns are actual class
    # size 10x10
    confusion_matrix = lra.test_multiclass(optimal_weights_of_all_classes)

    # Plot the confusion matrix and confusion table of each class
    lra.plot_confusion_matrix_and_tables(confusion_matrix=confusion_matrix, test_results=test_results)


if __name__ == '__main__':
    # Entry point of the script
    main(save_weights=False, use_weights_from_file=False)
