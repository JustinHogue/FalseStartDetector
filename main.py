import numpy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from matplotlib import pyplot


def main():
    # We create the X dataset where the first column is the time reaction in seconds and 
    # the second is the intensity of the EMG in mV
    X = numpy.random.rand(750,2)

    # We build the y classification
    y = []
    for x in X:
        if x[0] <= 0.1 and 0.2 <= x[1]:
            y.append(1)
        else:
            y.append(0)
    y = numpy.array(y)

    # We split the dataset matrices into random train and test subsets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, random_state=42)

    # We train the Perceptron model
    clf = QuadraticDiscriminantAnalysis()
    clf = clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))

    # Create a grid here to display the decision regions for each classifier
    h = 0.02 # not too small step size
    minimum = numpy.argmin(X, axis=0)
    x_min = X[minimum[0]][0]
    y_min = X[minimum[1]][1]
    maximum = numpy.argmax(X, axis=0)
    x_max = X[maximum[0]][0]
    y_max = X[maximum[1]][1]
    xxgrid, yygrid = numpy.meshgrid(numpy.arange(x_min - h, x_max + h, h), numpy.arange(y_min - h, y_max + h, h))

    # We initialize the pyplot figure
    fig = pyplot.figure()
    ax = fig.add_subplot(111)

    # We will analyze the entire grid and draw it according to its classification
    Z = clf.predict(numpy.c_[xxgrid.ravel(), yygrid.ravel()])
    Z = Z.reshape(xxgrid.shape)
    ax.contourf(xxgrid, yygrid, Z, cmap='Paired_r', alpha=0.75)
    ax.contour(xxgrid, yygrid, Z, colors='k', linewidths=0.1)
    ax.scatter(X[:,0], X[:,1], c=y, cmap='Paired_r', edgecolors='k')

    # We show the final result
    pyplot.show()


if __name__ == "__main__":
    main()