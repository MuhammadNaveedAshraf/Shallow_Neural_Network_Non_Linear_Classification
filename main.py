# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import matplotlib.pyplot as plt

from planar_utils import *
from model import *

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
    X,Y =load_planar_dataset()
    parameters = nn_model(X, Y, n_h=16, num_iterations=10000, print_cost=True)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    # plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    # plt.title("Decision Boundary for hidden layer size " + str(4))
    # plt.show()
    plot_decision_boundary1(X, Y,parameters) #(W1, b1, W2, b2))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.show()
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
