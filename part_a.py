import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
import urllib
import time


class PerceptronLearningAlgorithm(object):
    def __init__(self, inputs, labels, epochs=1, test_amount=10000):
        self.inputs = self.prepare_input(inputs)  # flattened data with 1 in the beginning, size inputs (number of images) x785
        # self.y = self.transform_labels_to_hot_vector(labels)  # matrix of hot vectors, size data (number of images) x 10
        self.labels = labels
        self.inputs_train, self.inputs_test, self.labels_train, self.labels_test = train_test_split(self.inputs, self.labels, test_size=test_amount)
        self.epochs = epochs

    def prepare_input(self, inputs):
        inputs = np.true_divide(inputs, 255.0)  # normalize all the inputs to be between 0-1
        return self.add_bias(inputs)

    def add_bias(self, inputs):
        """Adds bias to the inputs.
        The bias is constant and is equal to 1"""
        a, b = np.shape(inputs)
        c = np.ones((a, 1))
        return np.hstack((c, inputs))

    def transform_labels_to_hot_vector(self, labels):
        hot_vectors = np.zeros((len(labels), 10))  # column vector, each row will have a hot vector of a label
        for i in range(len(labels)):
            label = int(labels[i])
            hot_vectors[i, label] = 1
        return hot_vectors

    def plot_image(self, image, predicted_class=None, actual_class=None):
        """Plots one image out of the images
        Parameters: image - a vector of size 1x785 with normalized values (all values between 0-1)
        This function is just for debugging or visualizing, it is not used in calculations"""

        image = image[:, 1:]  # getting rid of the bias
        image = np.reshape(image, (28, 28))  # reshaping to size 288x28
        image = np.multiply(image, 255)  # Un-normalizing the image
        if predicted_class and actual_class:
            plt.xlabel("Predicted Class: {}\nActual Class: {}".format(predicted_class, actual_class))
        plt.imshow(image, cmap='gray')
        plt.show()

    def initial_weights(self, number_of_inputs):
        """Initial weights, returns a vector of size 785x1 of random numbers between 0-1.
        we subtract 0.5 because the weights can be negative."""
        return np.random.rand(number_of_inputs, 1) - 0.5

    def sign_function(self, weights, one_input):
        """Returns the sign of the dot product of weights and inputs.
        Parameters: weights - a vector of size 785x1
                    inputs - a vector of size 785x1"""
        return np.sign(np.dot(weights.T, one_input))

    def one_vs_all_labels(self, labels, number_to_classify):
        """One vs all means using a binary classifier with more than two available outputs
        The label of the wanted class will be 1 and the rest will be -1"""
        modified_labels = np.where(labels == number_to_classify, 1, -1)  # modified labels is now 1 or -1 depending on the number to classify
        return modified_labels

    def update_weights(self, weights, x, y):
        """Update weights using the update rule"""
        return weights + y * x

    def predict(self, weights, one_input):
        """Predicts the output of the input
        Parameters: weights - a vector of size 785x1
                    one_input - a vector of size 785x1"""
        prediction = self.sign_function(weights=weights, one_input=one_input)[0]
        return prediction

    def test_classifier(self, weights, number_to_classify, inputs=None, labels=None):
        """Test the classifier and returns the true positive, true negative, false positive and false negative
        Parameters: inputs - size of 785 x test_set_size which each column has a different picture
                    labels - size of 1 x test_set_size which each column has the label of the image
                    weights - a vector of size 785 x 1"""
        if inputs is None:
            inputs = self.inputs_test.T  # inputs is now in size of 785x10000 each column is an image
        if labels is None:
            labels = self.one_vs_all_labels(self.labels_test, number_to_classify=number_to_classify)  # size 1 x 10000

        true_pos = 0
        true_neg = 0
        false_pos = 0
        false_neg = 0

        test_set_size = inputs.shape[1]  # number of images in the test set
        for i in range(test_set_size):  # iterate through all images
            current_input = np.reshape(inputs[:, i], (785, 1))
            current_label = labels[i]
            prediction = self.predict(weights=weights, one_input=current_input)  # get the estimator

            if current_label == 1:
                if prediction == 1:
                    true_pos += 1
                if prediction == -1:
                    false_neg += 1

            if current_label == -1:
                if prediction == 1:
                    false_pos += 1
                if prediction == -1:
                    true_neg += 1

        return true_pos, true_neg, false_pos, false_neg

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
        # fig, ax = plt.subplots()
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

    def calculate_error(self, inputs, labels, weights):
        """Return the in-sample error
        This error is defined by the number of misclassified examples divided by the number of examples"""
        train_set_size = inputs.shape[1]  # number of images in the training set
        wrong_predictions_counter = 0
        for i in range(train_set_size):  # iterate through all images
            current_input = np.reshape(inputs[:, i], (785, 1))
            current_label = labels[i]
            prediction = self.predict(weights=weights, one_input=current_input)  # get the estimator

            # if the prediction is wrong then add to counter of wrong predictions
            if prediction != current_label:
                wrong_predictions_counter += 1

        return wrong_predictions_counter / train_set_size

    def train_binary_classifier(self, number_to_classify, epochs=None):
        """Trains one binary class out of all the 10 multi-class PLA
        number_to_clasify is the number that the classifier will train in one vs all method"""
        if epochs is None:
            epochs = self.epochs

        inputs = self.inputs_train.T  # inputs is now in size of 785x60000 each column is a picture
        labels = self.one_vs_all_labels(self.labels_train, number_to_classify)  # size 1x60000, labels are now 1 or -1
        train_set_size = inputs.shape[1]  # number of images in the training set
        weights = self.initial_weights(inputs.shape[0])  # weights is size 785x1

        pocket = weights  # for pocket algorithm
        pocket_error = self.calculate_error(inputs=inputs, labels=labels, weights=pocket)

        start_time = time.time()
        for _ in range(epochs):  # go through all training set epoch times
            for i in range(train_set_size):  # iterate through all images
                current_input = np.reshape(inputs[:,i], (785, 1))
                current_label = labels[i]
                prediction = self.predict(weights=weights, one_input=current_input)  # get the estimator

                # if the prediction is wrong then update the weights
                if prediction != current_label:
                    weights = self.update_weights(weights, current_input, current_label)

                # calculating error and updating the pocket
                if (i+1) % 1000 == 0:
                    error = self.calculate_error(inputs=inputs, labels=labels, weights=weights)
                    if error < pocket_error:
                        pocket_error = error
                        pocket = weights
        elapsed_time = time.time() - start_time
        print("Finished training class {}\nThe training took {:0.3f} seconds\n"
              .format(number_to_classify, elapsed_time))
        return pocket

    def train_all_classes(self, epochs=None):
        """Trains all 10 classes as part of multi-class PLA
        Returns an array of size 785x10 which each column is the optimal weights for a class.
        e.g returns np.array([w0, w1, ..., w9]) which each w is size 785x1"""

        print("Started training, this will take a few minutes...")
        if epochs is None:
            epochs = self.epochs

        start_time = time.time()

        weights = None
        for i in range(10):
            # Find the optimal weights of each class (weights i is size 785x1)
            weights_i = self.train_binary_classifier(number_to_classify=i, epochs=epochs)

            # save the optimal weights to form an array of optimal weights of each class
            if i == 0:
                weights = weights_i
            else:
                weights = np.hstack((weights, weights_i))

        elapsed_time = time.time() - start_time
        print("*********\nFinished Training all classes.\nThe training took {:0.3f} seconds".format(elapsed_time))

        return weights

    def test_all_classes(self, weights):
        """Test all the classes of the multi-class PLA
        Parameters: weights - an array of size 785x10, each column has the optimal weights of each class
                    like the returned value of train_all_classes"""

        # resulting_test... is in format [[true_pos0, true_neg0, false_pos0, false_neg0], [true_pos1, true_neg1, false_pos1, false_neg1],...]
        resulting_test_of_all_classes = []
        for i in range(10):
            weights_i = np.reshape(weights[:,i], (785, 1))
            true_pos, true_neg, false_pos, false_neg = self.test_classifier(weights_i, i)
            resulting_test_of_all_classes.append([true_pos, true_neg, false_pos, false_neg])

        return resulting_test_of_all_classes

    def plot_confusion_table_of_all_classes(self, test_results, show_table=False):

        for i in range(10):
            true_pos, true_neg, false_pos, false_neg = test_results[i]
            self.plot_confusion_table(true_pos=true_pos, true_neg=true_neg, false_pos=false_pos,
                                      false_neg=false_neg, number_to_classify=i)
        if show_table:
            plt.show()

    def predict_multiclass(self, weights, one_input):
        """Predicts the output of the multi-class PLA
        Parameters: weights - a vector of size 785x10 which is column is the optimal weights of each class
                    one_input - a vector of size 785x1 (one image)"""

        predicted_labels = []

        for i in range(10):  # iterate through all optimal weights
            current_weights = np.reshape(weights[:,i], (785, 1))  # take weights i
            prediction = np.dot(current_weights.T, one_input)  # and get the dot product with the input
            predicted_labels.append(prediction)  # save the result

        predictions_array = np.array(predicted_labels)
        return np.argmax(predictions_array)  # the predicted label is the argmax of the labels predicted by each optimal weights

    def test_multiclass(self, weights, inputs=None, labels=None):
        """Test the classifier and returns confusion matrix
        Parameters: inputs - size of 785 x 10000 which each column has a different picture
        labels - size of 1 x 10000 which each column has the label of the image
        weights - a vector of size 785 x 10 which is column has the optimal weights of each class"""

        if inputs is None:
            inputs = self.inputs_test.T  # inputs is now in size of 785x10000 each column is an image
        if labels is None:
            labels = self.labels_test  # size 1 x 10000

        # initialize the confusion matrix to zeros
        # rows are predicted class and columns are actual class
        confusion_matrix = np.zeros((10,10), dtype="int32")

        test_set_size = inputs.shape[1]  # number of images in the test set
        for i in range(test_set_size):
            current_input = np.reshape(inputs[:, i], (785, 1))
            current_label = int(labels[i])
            prediction = self.predict_multiclass(weights=weights, one_input=current_input)
            confusion_matrix[prediction][current_label] += 1

        return confusion_matrix

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
        plt.title("Confusion Matrix of the Multi-Class PLA\n")
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

    def save_weights(self, weights, path=r"./multiclass_weights_part_a.npy"):
        """Save weights for further use"""
        np.save(path, weights)

    def get_weights_from_file(self, path=r"./multiclass_weights_part_a.npy"):
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

    # instantiate a class of the PLA with the mnist data set
    pla = PerceptronLearningAlgorithm(inputs=mnist["data"], labels=mnist["target"], epochs=1)

    # Train all the 10 classes of the multi-class PLA
    # 10 classes, one for each digit of the mnist data set
    # Training is on the training set which is 60,000 images
    if use_weights_from_file:
        optimal_weights_of_all_classes = pla.get_weights_from_file()
    else:
        optimal_weights_of_all_classes = pla.train_all_classes()  # size 785x10 which each column is the optimal weights for a class.

    if save_weights:
        pla.save_weights(weights=optimal_weights_of_all_classes)

    # Testing each separate class
    # Testing is on the testing set which is 10,000 images
    test_results = pla.test_all_classes(optimal_weights_of_all_classes)  # See pla.test_all_classes_part_a for return value

    # Get the confusion matrix
    # Rows are predicted class and columns are actual class
    # size 10x10
    confusion_matrix = pla.test_multiclass(optimal_weights_of_all_classes)

    # Plot the confusion matrix and confusion table of each class
    pla.plot_confusion_matrix_and_tables(confusion_matrix=confusion_matrix, test_results=test_results)

if __name__ == '__main__':
    # Entry point of the script
    main(save_weights=False, use_weights_from_file=False)
