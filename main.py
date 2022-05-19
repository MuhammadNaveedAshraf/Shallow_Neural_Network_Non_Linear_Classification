# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import matplotlib.pyplot as plt

from planar_utils import *
from model import *

def classify():
    X,Y =load_planar_dataset()
    parameters,costs = nn_model(X, Y, n_h=16, num_iterations=10000, print_cost=True)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    plt.title("Decision Boundary for hidden layer size " + str(16))
    # plt.show()
    # plot_decision_boundary1(X, Y,parameters) #(W1, b1, W2, b2))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.show()
def try_learning_rates(X, Y , learning_rates):
    models = {}
    for i in learning_rates:
        print("learning rate is: " + str(i))
        models[str(i)] = nn_model(X, Y, n_h=16, num_iterations=10000, learning_rate=i, print_cost=False )
    for i in learning_rates:
        plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(i))
    plt.ylabel('cost')
    plt.xlabel('iterations(hundreds)')
    legend = plt.legend(loc='upper center', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    plt.show()
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #classify()

    X, Y = load_planar_dataset()
    learning_rates=[0.1,1,2,3]
    try_learning_rates(X, Y, learning_rates)
