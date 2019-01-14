import numpy as np
import random 
from datetime import datetime
random.seed(datetime.now())

def activate(h):
    return 1/(1 + np.exp(-h))

#Inputs with their respective biases
x = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
y = np.array([0,1,1,0])
#Randomize the weight matrix
w = 2 * np.random.rand(3,2) - 1
w2 = 2* np.random.rand(3,1) - 1

#learning rate or (eta) 
learn = .5

j = 0

while (j < 1000000): 

    flag = 0 
    for i in range(x.shape[0]):
        #Select a singular input from the list
        input = x[i]
        input = np.reshape(input, (1,3))

        #Feedforward 
        h = np.dot(input, w)
        hidden = activate(h)
        hidden = np.append(hidden, [1])
        hidden = np.reshape(hidden, (1,3))
        output = np.dot(hidden,w2)
        result = activate(output)
        result = result[0]
        r = result

        if result > 0.9:
            true = 1
        elif  result < 0.1:
            true = 0
        else:
            true = 3

        if true == y[i]:
            flag += 1
        else:
            flag = 0

        #Compute the backpropagation using the derivative of 
        #the sigmoid and the error from the actual value

        d2 = ((1 - result) * result) * (y[i] - result)
        d1 = (hidden * (1 - hidden)) * d2.dot(w2.T)
        d2 = np.reshape(d2, (1,1))

        update2 = (hidden.T.dot(d2))
        update1 = (input.T.dot(d1[:,:-1]))

        #Update the weights

        w2 += learn * update2
        w += learn * update1
    
        print(str(input[0][0]) + " " + str(input[0][1]))
        print("result " + str(r[0]))
        
        j = j + 1
    
    if flag == 4:
        print j
        break

print w
print w2
