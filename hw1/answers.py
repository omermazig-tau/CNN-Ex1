r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**

Increasing k may lead to improved generalization for unseen data up to a certain point. However, beyond that point, increasing k may result in decreased performance.
When k is small, the algorithm can overfit the training data, meaning it may not perform well on unseen data.
On the other hand, when k is too large, the algorithm may underfit the training data, meaning it may not capture the complexity of the underlying patterns in the data.
When k=1, the algorithm only considers the closest data point, and this can result in overfitting.
On the other hand, when k=N (N is the size of the dataset), the algorithm considers all the data points, and this can result in underfitting because we will label all images the same, according to the label that appears the most times in the training set.
Therefore, the optimal value of k is between these two extremes.
"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**
The selection of Delta is arbitrary because the Lambda parameter has the same purpose as Delta, because they both affect the size of the weight matrix.
Delta can be interpreted as a margin offset that can be adjusted to increase or decrease the weights of the SVM (If there is a sample that crosses the margin than the loss is larger). 
For any Delta value we can use larger values of W to increase the difference between the scores so the loss will be smaller.
However, the regularization term already imposes a penalty on large weights, which affects the size of the weight matrix W (If the weights are large the regularizatoin term in the loss is larger).
So for any Delta value we can control the size of W with Lambda value. That is why we can change the value of Lambda and to get the same result as changing the value of Delta.
"""

part3_q2 = r"""
**Your answer:**
1. The linear model learns to assign high scores to images where the white and black pixels are located in the same areas as in the data points labeled with a specific class in the training set.
This is reflected in the visualization of the weights, where bright colors indicate areas where the model expects to have white pixels, and dark colors indicate areas where the model expects to see black pixels.
To interpret the classification errors, we can examine images where the model made incorrect predictions:
These images often have digits that are not in an aligned angle or have additional or missing lines, causing their white or black areas to overlap with a different class of digits.
So we can see that the model misclassifies images when the arrangement of white and black pixels does not match the patterns it learned during training.

2.  
similarity:
In KNN we take from the train dataset the K closest images in the pixel space,
and we can see in the visualization section that in the linear classifier the images that represent each class's classifier are simillar in the pixel space to the images in that class. 

differences:
In KNN we take from the train dataset the K closest images in the pixel space and we predict the class of the image according to that.
In the linear classifier we divide the input space between classes and classifying based on which side of the hyperplane an unclassified image lands when placed in the input space. This is done by calulating the dot product score and choosing the class with the highest score.

In addition, in knn we predict the class based on the k closest neighbors.
in the linear classifier, we predict the class based on all the images in the train dataset.
"""

part3_q3 = r"""
**Your answer:**
1.The learning rate is good.
If the learning rate was too high, the classifier would miss the minimum of the loss function and it the graph would going up and down (and not going down monotonivally).
If the learning rate was too low, the graph would be less curved and we would see it descend more slowly.
2. The model is slightly overfitted to the training set.
This is because the training accuracy is a slightly higher than the validation accuracy. 
"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
**Your answer:**
The ideal pattern to see in a residual plot is y-y_hat=0 : an horizontal line which means that the model is more accurate.
After adding non-linear layers we get better results comparing to the top-5 features.
We can see that the residual plot is closer to the ideal pattern. We can see also that also that the R-square value is better (closer to one). 
"""

part4_q2 = r"""
**Your answer:**
The model fitted the data 180 times in total because we have 3 folds and 3 degrees and 20 lambdas. So in total we have:
3 * 3 * 20 = 180 times.
"""

# ==============
