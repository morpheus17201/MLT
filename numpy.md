# NUMPY Notes from MLT Week 0

## Create Numpy Arrays

```python
#import numpy as np
arr = np.array([1, 2, 3]) #passing pythong list as array
#creates matrix from the nested list
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])
arr = np.zeros(4) # creates 1d array with 4 zeros => array([0., 0., 0., 0. ])
arr = np.zeros((4,2)) # creates 4 rows x 2 cols matrix of zeros !!Note the argument is a tuple, not just comma separated list
arr = np.ones(4) # creates 1d array with 4 ones => array([1., 1., 1., 1. ])
arr = np.ones((4,2)) # Same as above - creates 4 x 2 matrix of ones
arr = np.eye(3) #Creates 3x3 identity matrix
```

The numpy arrays are vectors. e.g. the np.array([1, 2, 3]) creates 1d (column?) vector

## Shape of array

To get shape of vector

```python
# import numpy as np

arr = np.array([1, 2, 3]) #passing pythong list as array
arr.shape #shape is variable, not a member
## the above command will output (3,)
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

arr.shape # will output (3,3)

#The following relationship always holds good
arr.shape[0] == len(arr) # will output True
```

## Addition, Multiplication and Functions

```python
#import numpy as np

#multiplication by scalar: works for both vectors and matrices
x = np.array([1, 2, 3])
y = 3 * x

# Elementwise addition of two arrays: Works for both vectors and matrices
z = x + y

# Elementwise multiplication of two arrays ("Hadamard Product")
# Works for both vectors and matrices
z = x * y

# Dot product of two vectors
z = np.dot(x, y)

# Product of two matrices
Z = A @ B

# Applying function to each element to get new array

#Applying function directly: Works for both vectors and matrices
z = x ** 2 # squares each element

# Taking log of each element: Works for both vectors and matrices
z = np.log(x)

# Transpose of a matrix
M = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

# Two ways to take transpose
# 1) Use attribute e.g.
M_T = M.T

#2) Use Numpy function
M_T = np.transpose(M)

```

## Product of a matrix and a vector

This is very similar to the product of two matrices. Given the matrix $\mathbf{A}$ and the vector $\mathbf{x}$:

$$
\textbf{A} = \begin{bmatrix}
1 & 2 & 3\\
4 & 5 & 6\\
7 & 8 & 9
\end{bmatrix}, \textbf{x} = \begin{bmatrix}
6\\
7\\
8
\end{bmatrix}
$$

The product $\mathbf{Ax}$ is given by:

$$
\mathbf{C} = \mathbf{A x} = \begin{bmatrix}
44\\
107\\
170
\end{bmatrix}
$$
