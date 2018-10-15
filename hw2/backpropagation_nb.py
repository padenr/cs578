import numpy as np

def activate(h):
    return 1.0/(1.0 + np.exp(-h))

x = np.array([[0,0],[0,1],[1,0],[1,1]], float)
y = np.array([0,1,1,0])
w = 2* np.random.rand(2,2) - 1
w2 =2* np.random.rand(2,1) - 1
learn = .9

print(x)

print(" V1   V2   t3   y1   y2   y3   V3    e1     e2     e3   W1   W2   W3   W4   W5    W6    b1    b2    b3")

j = 0

while (j < 10000): 

    #0 and 0
    print(w)
    print(w2)
    for i in range(x.shape[0]):
        input = x[i]
        input = np.reshape(input, (1,2))
        h = np.dot(x[i], w)
        hidden = np.zeros((1,2))
        hidden = activate(h)
        output = np.dot(hidden,w2)
        result = activate(output)
        result = result[0]
        r = result

        

        d2 = (result *(1 - result)) * (y[i] - result)
        d1 = (hidden * (1 - hidden)) * (np.dot(d2,np.transpose(w2)))
        hidden = np.reshape(hidden,(1,2))
        d2 = np.reshape(d2, (1,1))
        dT2 = np.dot(np.transpose(hidden),d2)
        w2 += (-learn * dT2)

        dT1 = np.dot(np.transpose(input),d1[:,:-1])
        w += (-learn * dT1)
        w += (-learn * dT1)
        w += (-learn * dT1)
    
        #w1,w2,b1,b2,o5,e,e3,e4 = shift_network(x,hidden,w1,w2,b1,b2,output,0)
        #result1 = o5
        print(x[i])
        print("result " + str(r))

        #print_all(x1, 0, h, output, o5, e, e3, e4, w1, w2, b1, b2)

        j = j + 1
