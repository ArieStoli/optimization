import scipy.sparse as sp
import scipy.io


def read_matrix(filename):
    mat = scipy.io.loadmat(filename, squeeze_me=True)['X3']
    return mat


def derive_x(X):
    n, m = X.shape
    x = X.reshape(-1, 1, order='F')
    A = sp.lil_matrix((n * m, n * m), dtype=int)
    for i in range(n * (m - 1)):
        A[i, i] = -1
        A[i, i + n] = 1
    return A


def derive_y(X):
    n, m = X.shape
    x = X.reshape(-1, 1, order='F')
    A = sp.lil_matrix((n * m, n * m), dtype=int)
    for i in range(n * m):
        if i in range(n-1, n * m, n):
            continue
        A[i, i] = -1
        A[i, i + 1] = 1
    return A


if __name__ == '__main__':
    X = read_matrix('X3.mat')
    x = X.reshape(-1, 1, order='F')
    A_x = derive_x(X)
    A_y = derive_y(X)
    y_hat_x = A_x.dot(x)
    y_hat_y = A_y.dot(x)

    n, m = X.shape
    # all rows are different than 0 except rows which multiply by n
    # (index is minus 1 because the array starts at 0)
    print(y_hat_y[range(n-2,n*m,n)]) # should be different than 0
    print("______________________________")
    print(y_hat_y[range(n-1,n*m,n)]) # should be 0