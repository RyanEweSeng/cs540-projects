from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

n = 2414 # number of images
d = 1024 # dimension of an image 32x32

def load_and_center_dataset(filename):
    # load the dataset from the .npy file
    dataset = np.load(filename)

    # center the dataset around its origin
    centered_dataset = dataset - np.mean(dataset, axis=0)

    # return as a numpy array of floats
    return centered_dataset

def get_covariance(dataset):
    # calculate the covariance matrix of the dataset
    transposed_dataset = np.transpose(dataset)
    dot_prod = np.dot(transposed_dataset, dataset)
    cov_mat = dot_prod / (n-1)

    return cov_mat

def get_eig(S, m):
    # perform eigendecomposition
    s_len = len(S)
    w, v = eigh(S, subset_by_index=[s_len-m, s_len-1])
    
    # diagonal mat with largest m eigenvalues in descending order
    w = np.diag(w[::-1])

    # mat with corresponding eigenvectors as columns
    v = np.fliplr(v)

    return w, v

def get_eig_prop(S, prop):
    # perform eigendecomposition to get total eigenvalues
    w = eigh(S, eigvals_only=True)
   
    # find all eigenvalues that are greater than the prop
    a = []
    sum_of_all_eigenvalues = np.sum(w)
    for i in range(len(S)):
        if w[i] / sum_of_all_eigenvalues > prop:
            a.append(w[i])

    # find the corresponding eigenvectors
    x, y = eigh(S, subset_by_value=[a[0]-1, a[len(a)-1]+1])

    a = np.diag(a[::-1])
    b = np.fliplr(y)

    return a, b

def project_image(image, U):
    # compute weights
    weights = np.dot(image, U)

    # calculate pca
    x_pca = np.dot(weights, np.transpose(U))

    return x_pca

def display_image(orig, proj):
    # reshape the images to 32x32
    orig_plot = np.reshape(orig, [32, 32])
    proj_plot = np.reshape(proj, [32, 32])

    # create a figure w/ one row of two subplots
    fig, (subplot1, subplot2) = plt.subplots(nrows=1, ncols=2, figsize=(20,10))

    # title the subplots
    subplot1.set_title('Original')
    subplot2.set_title('Projection')

    # call imshow
    val1 = subplot1.imshow(orig_plot, aspect='equal')
    val2 = subplot2.imshow(proj_plot, aspect='equal')

    # render
    fig.colorbar(val1, ax=subplot1)
    fig.colorbar(val2, ax=subplot2)
    plt.show()

x = load_and_center_dataset("YaleB_32x32.npy")
S = get_covariance(x)
Lambda, U = get_eig(S, 2)
Lambda_prop, U_prop = get_eig_prop(S, 0.07)
projection = project_image(x[0], U)
display_image(x[0], projection)

