import part_b
from tkinter import *
import numpy as np
import random

class Paint(object):
    DEFAULT_PEN_SIZE = 5.0
    DEFAULT_COLOR = 'black'

    def __init__(self):
        self.root = Tk()

        # canvas widget
        self.c = Canvas(self.root, bg='white', width=270, height=270)
        # self.c = Canvas(self.root, bg='white', width=28, height=28)
        self.c.grid(row=0, columnspan=5)

        # go button
        self.go_button = Button(self.root, text="go", command=self.go)
        self.go_button.grid(row=1, column=0)

        # reset button
        self.reset_button = Button(self.root, text="reset", command=self.erase)
        self.reset_button.grid(row=1, column=4)

        self.instructions = Label(self.root, text="Draw a digit (0-9) and press go")
        self.instructions.grid(row=1, column=2)

        self.prediction = StringVar()  # when this variable is changed the prediction label is automatically changed is well
        self.prediction_label = Label(self.root, text="", textvariable=self.prediction, justify="center")
        self.prediction_label.grid(row=2, column=2)

        self.old_x = None
        self.old_y = None

        self.line_width = 10
        self.color = "black"

        self.c.bind('<B1-Motion>', self.paint) # when you press the mouse on the canvas this happens
        self.c.bind('<ButtonRelease-1>', self.reset)  # when you release the mouse this happens

        # run the app
        self.root.mainloop()


    def paint(self, event):
        """Paint on the canvas"""
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=self.color,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)

        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        """Stop Painting"""
        self.old_x, self.old_y = None, None

    def go(self):
        """Predicts the number that has been drawn using the softmax regression"""
        # We deliberately made the canvas 10 times bigger, so now we scale it down
        self.c.scale(ALL, 0, 0, 0.1, 0.1) # parameters: tagOrId xOrigin yOrigin xScale yScale
        pixels = self.c.find_all()  # get all he pixels
        coords = [self.c.coords(pixel) for pixel in pixels]  # get the coordinates of each pixel
        prediction, confidence = predict(coords)  # use the machine learning model to predict what number is the drawing
        text = "The number is: {}".format(prediction)
        self.prediction.set(text)  # update the prediction label
        self.c.scale(ALL, 0, 0, 10, 10)  # scale back so the change won't be visible

    def erase(self):
        """Erases everything from the canvas"""
        self.c.delete(ALL)
        self.prediction.set("")  # reset the prediction label

def predict(coords):
    """Predicts what digit was drawn on the canvas
    Returns a string representing the digit"""
    image = turn_coords_to_image(coords)
    lra = part_b.LogisticRegressionAlgorithm(inputs=image, labels=None)
    weights = lra.get_weights_from_file(path=r"./multiclass_weights_part_b_10000_iterations.npy")
    prediction = lra.predict_multiclass(weights=weights, inputs=image)  # shape (10, 1)
    prediction_class = lra.classify_multiclass(predictions=prediction)
    confidence = prediction[prediction_class]
    return prediction_class, confidence

def turn_coords_to_image(coords):
    """Translates the coordinates of the drawing to a numpy array with shape (784, ) of black and white.
    This is so it will match the machine learning model (mnist data set is images of 28x28 so flattened images are 784x1."""
    image = np.zeros((28, 28))  # generating a zero numpy array (meaning it is all black)
    white = 255
    coords = generate_coord_pairs(coords=coords)

    # color in white the parts that were drawn on the canvas
    for coord in coords:
        if 0 < coord[0] < 28 and 0 < coord[1] < 28:
            # color the pixel itself
            image[coord[1]][coord[0]] = white

    image = center_image(image)
    image = fill_adjecnt_pixels(image=image, color='white')
    # image = fill_adjecnt_pixels(image=image, color='light gray')
    # image = fill_adjecnt_pixels(image=image, color='dark gray')


    # plt.imshow(image, interpolation='nearest', cmap='gray')
    # plt.show()

    image = np.reshape(image, (784, ))  # reshaping to fit the model

    return image

def random_dark_gray():
    return random.randint(50, 100)

def random_light_gray():
    return random.randint(150, 200)

def fill_adjecnt_pixels(image, color):
    """Fills pixels that are adjacent to already full pixel with a specified color"""
    white = 255
    i_s, j_s = np.where(image != 0)
    for i, j in zip(i_s, j_s):
        if i == 0 or i == 27:
            continue
        if j == 0 or j == 27:
            continue

        # if pixel contains a color then fill adjacent pixels if not already filled
        if not image[i - 1][j + 1]:
            image[i - 1][j + 1] = white if color == 'white' else (random_dark_gray() if 'dark' in color else random_light_gray())
        if not image[i + 0][j + 1]:
            image[i + 0][j + 1] = white if color == 'white' else (random_dark_gray() if 'dark' in color else random_light_gray())
        if not image[i + 1][j + 1]:
            image[i + 1][j + 1] = white if color == 'white' else (random_dark_gray() if 'dark' in color else random_light_gray())
        if not image[i - 1][j + 0]:
            image[i - 1][j + 0] = white if color == 'white' else (random_dark_gray() if 'dark' in color else random_light_gray())
        if not image[i + 1][j + 0]:
            image[i + 1][j + 0] = white if color == 'white' else (random_dark_gray() if 'dark' in color else random_light_gray())
        if not image[i - 1][j - 1]:
            image[i - 1][j - 1] = white if color == 'white' else (random_dark_gray() if 'dark' in color else random_light_gray())
        if not image[i + 0][j - 1]:
            image[i + 0][j - 1] = white if color == 'white' else (random_dark_gray() if 'dark' in color else random_light_gray())
        if not image[i + 1][j - 1]:
            image[i + 1][j - 1] = white if color == 'white' else (random_dark_gray() if 'dark' in color else random_light_gray())

    return image

def center_image(image):
    i_s, j_s = np.where(image != 0)
    min_i = min(i_s)
    max_i = max(i_s)
    min_j = min(j_s)
    max_j = max(j_s)

    if min_j <= 14:
        height_roll_amount = int((min_j + (28 - max_j)) / 2) - min_j
    else:
        height_roll_amount = int((min_j - max_j) / 2) - (min_j - 14)

    if min_i <= 14:
        width_roll_amount = int((min_i + (28- max_i)) / 2) - min_i
    else:
        width_roll_amount = int((min_i - max_i) / 2) - (min_i - 14)

    image = np.roll(image, height_roll_amount, axis=1)
    image = np.roll(image, width_roll_amount, axis=0)
    return image


def generate_coord_pairs(coords):
    """The coordinates are groups of 4, now we make them a group of two (x value and y value)
    Also round values to the closest integer"""
    rounded_coord_pairs = []
    for coord in coords:
        pair1 = [int(coord[0]), int(coord[1])]
        pair2 = [int(coord[2]), int(coord[3])]

        # append to list, don't have duplicates
        if pair1 not in rounded_coord_pairs:
            rounded_coord_pairs.append(pair1)
        if pair2 not in rounded_coord_pairs:
            rounded_coord_pairs.append(pair2)

    return rounded_coord_pairs


if __name__ == '__main__':
    Paint()

