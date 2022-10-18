import sys
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

# this program outputs a .jpg file of the plot
# save plot using plt.savefig("plot.jpg")

if __name__ == '__main__':
    filename = sys.argv[1]

    # Question 2
    dataframe = pd.read_csv(filename)
    axes = dataframe.plot(x="year", y="days", figsize=(20,15))
    plt.xlabel("Year")
    plt.ylabel("Number of frozen days")
    plt.savefig("plot.jpg")

    # Question 3a
    data = dataframe.to_numpy()
    X = []
    for i in range(len(data)):
        x_i = [1, data[i][0]]
        X.append(np.transpose(x_i))
    X = np.array(X)
    print("Q3a:")
    print(X)

    # Question 3b
    Y = []
    for i in range(len(data)):
        y_i = data[i][1]
        Y.append(y_i)
    Y = np.array(Y)
    print("Q3b:")
    print(Y)

    # Question 3c
    X_T = X.T
    Z = np.dot(X_T, X)
    print("Q3c:")
    print(Z)

    # Question 3d
    I = np.linalg.inv(Z)
    print("Q3d:")
    print(I)

    # Question 3e
    PI = np.dot(I, X_T)
    print("Q3e:")
    print(PI)

    # Question 3f
    hat_beta = np.dot(PI, Y)
    print("Q3f:")
    print(hat_beta)

    # Question 4
    x_test = 2021
    y_test = hat_beta[0] + np.dot(hat_beta[1],x_test)
    print("Q4: " + str(y_test))

    # Question 5a
    sign = None
    if hat_beta[1] > 0:
        sign = ">"
    elif hat_beta[1] < 0:
        sign = "<"
    else:
        sign = "="
    print("Q5a: " + sign)

    # Question 5b
    print("Q5b: the sign represents the slope of the regression line; a < means it is a negative slope telling me that for an increase in years, there will be a decrease in the number of frozen days")

    # Question 6a
    x_star = (-hat_beta[0]) / hat_beta[1]
    print("Q6a: " + str(x_star))

    # Question 6b
    print("Q6b: since the regression slope is a small negative, it means there is a slow but gradual decrease in frozen days. Thus, an estimate of 400 years seems like a plausible estimate of when there will be no ice")

