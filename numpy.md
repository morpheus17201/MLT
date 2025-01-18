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
arr2 = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

arr2.shape # will output (3,3)

#The following relationship always holds good
arr2.shape[0] == len(arr2) # will output True

#ndim function gives number of dimensions (as against the shape) e.g.
arr.ndim # will output 1
arr2.ndim # will output 2
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

This is achieved as below:

```python
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
x = np.array([6, 7, 8])
C = A @ x
```

## Reshaping

Arrays can be reshaped. e.g. starting from vector to change it into matrix or vice versa

```python
M = np.array([[1, 2, 3], [4, 5, 6]])
x = M.reshape(6) # This will give x=array([1, 2, 3, 4, 5, 6])

# We can let numpy calculate the last dimension by specifying it as -1
M = np.array([[1, 2, 3], [4, 5, 6]])
P = M.reshape((3, -1)) # In this case, we are telling numpy that we need 3 rows. -1 as last argument specifies numpy should determine number of columns on its own
# The above command will make P = array([[1, 2],
#       [3, 4],
#       [5, 6]])
```

## Broadcasting and Matrix vector addition

In many ML models, we would have to add a vector to each row or column of a matrix. For example, consider the following case for row-wise addition:

### Row-wise addition

$$
\mathbf{M} = \begin{bmatrix}
1 & 2 & 3 & 4\\
5 & 6 & 7 & 8
\end{bmatrix}, \mathbf{b} = \begin{bmatrix}
1 & 1 & 1 & 1
\end{bmatrix}
$$

This is slight abuse of notation as we can't add a matrix and a vector together. However, the context often makes this clear:

$$
\mathbf{M} + \mathbf{b} = \begin{bmatrix}
2 & 3 & 4 & 5\\
6 & 7 & 8 & 9
\end{bmatrix}
$$

In numpy this becomes

```python
M = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
b = np.array([1, 1, 1, 1])
print('Shape of M:', M.shape)
print('Shape of b:', b.shape)

M + b
```

Thus, if we want to do row wise addition (i.e. add a vector to each row of matrix), then simple addition will work as numpy will 'broadcast' the vector to each row

### Column-wise addition

The technique above doesn't work for column-wise addition as numpy cannot broadcast to different shape as explained below. e.g. the below example doesn't work:


$$
\mathbf{M} = \begin{bmatrix}
1 & 2 & 3 & 4\\
5 & 6 & 7 & 8
\end{bmatrix}, \mathbf{b} = \begin{bmatrix}
1\\
2
\end{bmatrix}
$$

In the case, we have:

$$
\mathbf{M} + \mathbf{b} = \begin{bmatrix}
2 & 3 & 4 & 5\\
7 & 8 & 9 & 10
\end{bmatrix}
$$

**Why??** When two arrays of different dimensions are combined together using an arithmetic operation such as `+`, `NumPy` sees if it can **broadcast** them. This is best understood with images. Here is an example from the `NumPy` docs on row-wise addition:

![](https://numpy.org/doc/stable/_images/broadcasting_2.png)

**Source**: https://numpy.org/doc/stable/user/basics.broadcasting.html

For column-wise addition, simple addition doesn't work:

![](https://numpy.org/doc/stable/_images/broadcasting_3.png)

**Source**: https://numpy.org/doc/stable/user/basics.broadcasting.html

To solve for this, we need to use ```np.expand_dims``` function. This function expands the dimension of the array.

```python
b = np.array([1, 2])
# Shape of b is (2, ) i.e. 2 rows
b = np.expand_dims(b, 1)
# Shape of b becomes (2, 1) and b = [[1],
#                                    [2]]
```

Arguments for np.expand_dims

- The first argument to np.expand_dims is the array for which the dimensions need to be expanded.
- The second argument is the axis along which to increase, with 0 indexing. For a 2 dimensional matrix, dimension '0' means expand along rows and axis '1' means expand along columns (i.e. add empty columns)

Thus, for adding a column vector to the matrix, we should use:

```python
import numpy as np
M = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
b = np.array([1, 2])
print('Shape of M:', M.shape)
print('Shape of b before adding dimension:', b.shape)
b = np.expand_dims(b, 1)
print('Shape of b after adding dimension:', b.shape)
M + b
# this will give
# [[2, 3, 4, 5],
#  [7, 8, 9, 10]]
```

## Indexing and slicing an array

The indexing for the array works in the same way for as that for a standard python list. Examples below:

```python
M = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
# Get third row
M[2] #array([5, 6])

#Get second column
M[:, 1] #array([ 2,  4,  6,  8, 10])

#Get third and fourth rows
M[2:4] #array([[5, 6],
#              [7, 8]])

#Get 2nd and 3rd elements in the first column
M[1: 3, 0] #array([3, 5])

```

## Stacking of arrays

Sometimes, we would want to stack arrays. Consider the two matrices:

$$
\mathbf{A} =
\begin{bmatrix}
1 & 2\\
3 & 4
\end{bmatrix},
\mathbf{B} =
\begin{bmatrix}
5 & 6\\
7 & 8
\end{bmatrix}
$$

There are two ways to stack these two matrices:

### Row-wise

We could stack the two matrices along the rows, $\mathbf{A}$ on top of $\mathbf{B}$:

$$
\mathbf{C} =
\begin{bmatrix}
1 & 2\\
3 & 4\\
5 & 6\\
7 & 8
\end{bmatrix}
$$

This would be done as follows:
```python
C = np.concatenate((A, B), axis = 0) # axis=0 indicates rows in 2d matrix
```

### Column-wise

We could stack the two matrices along the columns, $\mathbf{A}$ to the left of $\mathbf{B}$:

$$
\mathbf{C} =
\begin{bmatrix}
1 & 2 & 5 & 6\\
3 & 4 & 7 & 8\\
\end{bmatrix}
$$

This would be done as follows:

```python
C = np.concatenate((A, B), axis = 1) # axis = 1 indicates columns in 2d matrix
```

## SUM, MEAN and VARIANCE

We can use np.sum() function or the ```.sum``` attribute to get the sum.
- **Sum of Rows:** i.e. sum of 1st row, sum of 2nd row, etc then we are summing along the column, hence we need to indicate axis as '1'. The resulting vector will have the same shape as number of rows in the matrix

$$
\mathbf{A} = \begin{bmatrix}
1 & 2 & 3 & 4\\
5 & 6 & 7 & 8
\end{bmatrix}
$$

The sum of the rows of the matrix is a vector:

$$
\text{rsum}(\mathbf{A}) = \begin{bmatrix}
10\\
26
\end{bmatrix}
$$

In `NumPy` this can be done as follows:

```python
A = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
row_sum = np.sum(A, axis=1) # outputs array([10,26])
# or
row_sum = A.sum(axis=1)

```

- **Sum of Columns:** i.e. sum of 1st col, sum of 2nd col, sum of 3rd col, etc. For getting the sum, we are summing across the rows, hence we need to indicate axis as 1. e.g.

```python
A = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
col_sum = np.sum(A, axis=0) # outputs array([6, 8, 10, 12])
# or
col_sum = A.sum(axis=0)
```
Just like sum, we can also find out mean and variance along rows or columns
