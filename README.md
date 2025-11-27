# Machine-Learning-in-Finance-Insurance-HS25

## Infos Project 2

m = training set
n = test set


2.
will have 2 different data sets from the 2 generating data funcions. They generate y_1 and y_2 from x

b) don't forget to work with the normalized data
(i) and (ii) are a single line of code only have to hand in the proability = True version
c) definitions are on the slides at the end in the logisitc regression slide deck

3.
(iii) For each individual i you need to sample 50k scenarios.

Need to invest more time in problem 3 , 1 & 2 fairly straight forward.


## Infos Project 3

Need to do everything from scratch but only using the demo code to get some ideas. Demos are with Pytorch


(i)Need to experiment with different NN architectures
(ii) H are 30 univariate NN and each one of them want's to learn to minimize the loss. All of the NN have different inputs. You can implement a for loop to calculate the all. Outputs are summed up in the loss. 

1.
a) Just check the solution --> check mathematically 
b) simulate
c) demo notebook on moodle course book which guides how to set up the NN which is on moodle
d) Once you have the price, then put it in the loss function and minimize the loss function
e) Histogram of losses centred around zero with very small variance (like 0.04) --> will have a non-zero loss over a certain interval as we can only hedge over discrete time instead of continous like in black-scholes model
f) (i) Just the derivative
  (ii) Plot one of theoretical model as well and compare. If there are differences then you need to train your model more (regularization, add layers, train longer, change activation function etc.)
g) ReLu, can be piceswise continuous. Some hedging strategies look much more like the analytical ones, find out why?

2. Hesta model has volatility process in the model as well. Minimize the loss over parameters of NN as well as the additonal parameter w.
a) We know the transition function of the process to simulate it. Shows how to implement the heston model
c) alpha is the level of risk aversion. Plot the hedge losses with histpgram and and show which is the more risk averse of them. The more risk averse you are the price should be higher. 


Want to replicate a bit the Deep Hedging which is shown in the Paper.


Deciding on number of layers:
Start with 16 neurons, then go up to 32. But everytime you add a layer the model gets more complex. So try with more neurons first before more layers. But for good performance might need 2 layers.
Activation
Hyperbolic tangent
Train for decent number of epochs: More than 10
Learning rate:
10^-3 works very well but if it learns too slowly then go to a higher learning rate otherwise go lower.

## Infos Project 4

1. Need to create a categorical feature out of the age for example and split it into buckets as we can see that it is not linear and therefore create buckets where we can approcimate that it is linear and then use the exp of the linear function on it. So one-hot encode the age data in it's individually assigned buckets. Give reasoning why we chose these intervals to create the categories.

2. Need to try out different architectures and will see that it is only marginally better then the Possion GLM. Not even 10% better

3. Make sure that the parameter which is chosen by the CV is not at the egdes as this means that the grid size needs to be enlarged. Check the variablility of the data and decide on a grid size and then choose that for the CV.
