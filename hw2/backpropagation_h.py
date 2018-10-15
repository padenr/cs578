import numpy as np

def activate(h):
    return 1/(1 + np.exp(-h))

#Inputs with their respective biases
x = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
y = np.array([0,1,1,0])
#Randomize the weight matrix
w = 2 * np.random.rand(3,2) - 1
w2 = 2* np.random.rand(3,1) - 1

#learning rate or (eta) 
learn = 0.5

j = 0

while (j < 6000): 

    #0 and 0
    #print("")
    for i in range(x.shape[0]):
        #Select a singular input from the list
        input = x[i]
        input = np.reshape(input, (1,3))

        if j == 1 or j == 2:
            print("input")  
            print(input)
            print(w)
        #Feedforward 
        h = np.dot(input, w)
        hidden = activate(h)

        if j == 1 or j == 2:
            print("hidden activation")
            print(h)
            print(hidden) 

        hidden = np.append(hidden, [1])
        hidden = np.reshape(hidden, (1,3))


        output = np.dot(hidden,w2)


        result = activate(output)

        
        if j == 1 or j == 2:
            print("output")
            print(w2) 
            print(output)
            print(result)


        if j == 1 or j == 2:
            print(result) 

        result = result[0]
        r = result

        #Compute the backpropagation using the derivative of 
        #the sigmoid and the error from the actual value

        d2 = ((1 - result) * result) * (y[i] - result)

        if j == 1 or j == 2:
            print("original deltas") 
            print(d2) 
            print(d1)
            
        d1 = (hidden * (1 - hidden)) * d2.dot(w2.T)
        d2 = np.reshape(d2, (1,1))

        update2 = (hidden.T.dot(d2))
        update1 = (input.T.dot(d1[:,:-1]))

        if j == 1 or j == 2:
            print("updates") 
            print(update2)
            print(update1)

        #Update the weights

        w2 += learn * update2
        w += learn * update1

        if j == 1 or j == 2:
            print("updated weights") 
            print(w2)
            print(w)
    
        #print(str(input[0][0]) + " " + str(input[0][1]))
        #print("result " + str(r[0]))
        
        j = j + 1
