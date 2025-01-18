# NUMPY Notes from MLT Week 0

## Create Numpy Arrays

```python
#import numpy as np
arr = np.array([1, 2, 3]) #passing pythong list as array
arr = np.zeros(4) # creates 1d array with 4 zeros => array([0., 0., 0., 0. ])
arr = np.ones(4) # creates 1d array with 4 ones => array([1., 1., 1., 1. ])
```

The numpy arrays are vectors. e.g. the np.array([1, 2, 3]) creates 1d (column?) vector

## Shape of array

To get shape of vector

```python
# import numpy as np

arr = np.array([1, 2, 3]) #passing pythong list as array
arr.shape #shape is variable, not a member
## the above command will output (3,)

#The following relationship always holds good
arr.shape[0] == len(arr) # will output True
```

## Addition, Multiplication and Functions

```python
#import numpy as np

#multiplication by scalar
x = np.array([1, 2, 3])
y = 3 * x

# Elementwise addition of two arrays
z = x + y

# Elementwise multiplication of two arrays ("Hadamard Product")
z = x * y

# Dot product of two vectors
z = np.dot(x, y)

# Applying function to each element to get new array

#Applying function directly
z = x ** 2 # squares each element

# Taking log of each element
z = np.log(x)

```
