# Video 1 and 2: Broad Paradigms of Machine Learning

* __Supervised learning__
  * Classication (Binary, Multiclass, Ordinal)
  * Regression
  * Ranking
  * Structure learning
* __Unsupervised learning:__ e.g. Automatically figuring out groups from data
  * Clustering
  * Representation learning: How do I represent data point in the best possible so 'something' becomes easier. e.g. image can be represented as something better than just pixel values. (Note: Representation learning can also be done in some way in supervised setting)
* __Sequential learning:__ Don't learn in one shot but learn in sequential fashion. Not all the data is given. Feedback is the form of supervision. Get some data, predict, then get feedback and correct using the feedback.
  * Online learning
  * Multi-armed Bandits
  * Reinforcement learning: mapping from state to action (policy). e.g. robot navigation to go from one end of the room to the other through obstacles

### Examples
    
|__Problem__|__Type of problem__|
|-------------------------|-----------------|
| Spam vs non-spam | Binary classification |
| Forecasting rainfall | Regression |
| Recommending movies | Ordinal classification |
| Friend suggestions | Link Prediction |
| Voice/instrument speration | Unsupervised learning |
| Grouping pictures in phone | Clustering |
| Stock market prediction | online problem |
| Robot navigation | Reinforcement learning |

### Smple example: Linear regression
 
 |__Element__|__Role__|__Branch of Mathematics__|
 |-------------|----------|---------------------------|
 | * __Structure__ | Need to decide if this is linear data | Linear Algebra |
 | * __Uncertainty__ | Probability/uncertainty | Probability |
 | * __Decisions__ | Convert data to decision (maximize or minimize) | Optimization |

# Video 3: Unsupervised learning (representation learning)

__Goal:__ Given a set of "data points", "understand" something "useful" about them.

* Datapoints: Vectors in d-dimensions $R_{d}$
* Understand: "Comprehension is compression" - George Chaitin.\
Comprehension - Understanding / learning

## Projection of a point on the line
To compress the dataset (i.e. to represent using fewer data points, if all the data points are on a single line, then we just need a scaling factors and one of the features for each of the data points and we can perefectly construct the data again.

However, if one of the points is not on the line passing through the other points, then we need to find the projection of that point on the line passing through the remaining points.
![image](https://github.com/user-attachments/assets/5ecc78f0-3439-4c9c-82ff-dc1eae671bee)

The projection of this outlier point 
$$
\begin{bmatrix}
x_{1}\\
x_{2}\\
\end{bmatrix}
