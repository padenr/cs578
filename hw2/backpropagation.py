import numpy as np

def activate(h):
    return 1.0/(1.0 + np.exp(-h))

x = np.array([[0,0],[0,1],[1,0],[1,1]], float)
y = np.array([0,1,1,0])
x = np.c_[x, np.ones(4)]
w = np.random.rand(3,2)
w2 = np.random.rand(3,1)
learn = 0.5

print(x)

print(" V1   V2   t3   y1   y2   y3   V3    e1     e2     e3   W1   W2   W3   W4   W5    W6    b1    b2    b3")

i = 0

while (i < 100): 

    #0 and 0
    for i in range(x.shape[0]):
        h = np.dot(x[i], w)
        hidden = activate(h)
        hidden = np.append(hidden, [1])
        print(hidden)
        output = np.dot(hidden,w2)
        result = activate(output)
        d2 = result * (1 - result) * (result - y[i])
        d1 = result * (1 - result) * (np.dot(d2,np.transpose(w2)))
        w2[0] += (-learn * result * d2)
        w2[1] += (-learn * result * d2)
        w2[2] += w2[2] + (-learn * d2)
        print(d1)
        w[0][0] += (-learn * hidden[0] * d1)
        w[1][0] += (-learn * hidden[0] * d1)
        w[2][0] += (-learn * d1)
        
        w[0][1] += (-learn * hidden[0] * d1)
        w[1][1] += (-learn * hidden[0] * d1)
        w[2][1] += (-learn * d1)
        #w1,w2,b1,b2,o5,e,e3,e4 = shift_network(x,hidden,w1,w2,b1,b2,output,0)
        #result1 = o5
        print(x[i])
        print(result)

        #print_all(x1, 0, h, output, o5, e, e3, e4, w1, w2, b1, b2)

        i = i + 1
