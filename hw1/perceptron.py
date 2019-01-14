import numpy as np


def shift_network_simple(nodes, w, b, output, target):

    if output >= 0: 
        result = 1
    else:  
        result = 0
    
    error = calc_error(target, result)

    w[0,0] = shift_weight(0.5, error, nodes[0,0], w[0,0])
    w[1,0] = shift_weight(0.5, error, nodes[0,1], w[1,0])
    b = change_bias(1, error, b)

    return (w, b, result, error)

def shift_network(nodes, hidden, weight1, weight2, bias1, bias2, output, target):

    if output >= 0: 
        result = 1
    else:  
        result = 0
    
    error = calc_error(target, result)
    error_3 = 0.5 * error * weight2[0,0]
    error_4 = 0.5 * error * weight2[1,0] 

    weight2[0,0] = shift_weight(0.5, error, hidden[0,0], weight2[0,0])
    weight2[1,0] = shift_weight(0.5, error, hidden[0,1], weight2[1,0])


    weight1[0,0] = shift_weight(0.5, error_3, nodes[0,0], weight1[0,0])
    weight1[1,0] = shift_weight(0.5, error_3, nodes[0,0], weight1[1,0])
    weight1[0,1] = shift_weight(0.5, error_4, nodes[0,1], weight1[0,1])
    weight1[1,1] = shift_weight(0.5, error_4, nodes[0,1], weight1[1,1])



    bias1[0,0] = change_bias(0.5, error_3, bias1[0,0])
    bias1[0,1] = change_bias(0.5, error_4, bias1[0,1])
    bias2 = change_bias(0.5, error, bias2)

    return (weight1, weight2, bias1, bias2, result, error, error_3, error_4)

def print_all_simple(nodes, target, output, result, error, weights, bias):
    print(" {}   {}   {}   {}   {}   {}   {}   {}    {}".format(nodes[0,0], nodes[0,1], target, output, result, error, weights[0,0], weights[1,0], bias))

def print_all(nodes, target, hidden, output, result, error, error_3, error_4, weight1, weight2, bias1, bias2): 
    print("{}   {}   {}    {}   {}   {}   {}   {}    {}   {}    {}   {}    {}     {}    {}    {}    {}    {}     {}".format(nodes[0,0], nodes[0,1], target, hidden[0,0], hidden[0,1], output, result, error, error_3, error_4, weight1[0,0], weight1[1,0], weight1[0,1], weight1[1,1], weight2[0,0], weight2[1,0], bias1[0,0], bias1[0,1], bias2))


def calc_error(target, output):
    return target - output

def shift_weight(alpha, error, node, weight):
    shift = alpha * error * node
    return weight + shift

def change_bias(alpha, error, bias):
    shift = alpha * error
    return bias + shift

def activate_hidden(h):
    
    if h[0,0] >= 0: 
        h[0,0] = .85
    else:  
        h[0,0] = .01

    if h[0,1] >= 0: 
        h[0,1] = .85
    else:  
        h[0,1] = .01

    return h



x1 = np.array([[0,0]], float)
x2 = np.array([[0,1]], float)
x3 = np.array([[1,0]], float)
x4 = np.array([[1,1]], float)
w = np.array([[0],[0]], float)

e = -1
b = -0.5
v3 = 1

i = 0

print(" V1   V2   t3   y3   V3    e   W1   W2    b")


while (i < 100): 

    #0 and 1 
    output = b + np.dot(x2,w)
    w,b,v3,e = shift_network_simple(x2,w,b,output, 1)
    print_all_simple(x2, 1, output, v3, e, w, b)

    #1 and 0
    output = b + np.dot(x3,w)
    w,b,v3,e = shift_network_simple(x3,w,b,output, 1)
    print_all_simple(x3, 1, output, v3, e, w, b)

    #1 and 1 
    output = b + np.dot(x4,w)
    w,b,v3,e = shift_network_simple(x4,w,b,output, 1)
    print_all_simple(x4, 1, output, v3, e, w, b)

    #0 and 0 
    output = b + np.dot(x1,w)
    w,b,v3,e = shift_network_simple(x1,w,b,output, 0)
    print_all_simple(x1, 0, output, v3, e, w, b)

    i = i + 1
    j = -1

while (True):

    j = j + 1 
    x1 = np.array([[0,0]], float)
    x2 = np.array([[0,1]], float)
    x3 = np.array([[1,0]], float)
    x4 = np.array([[1,1]], float)

    w1 = np.random.rand(2,2)
    w2 = np.random.rand(2,1)
    b1 = np.random.rand(1,2)
    b2 = 0

    print(" V1   V2   t3   y1   y2   y3   V3    e1     e2     e3   W1   W2   W3   W4   W5    W6    b1    b2    b3")

    i = 0

    while (i < 100): 

        #0 and 0
        h = b1 + np.dot(x1, w1)
        hidden = activate_hidden(h)
        output = b2 + np.dot(hidden,w2)
        w1,w2,b1,b2,o5,e,e3,e4 = shift_network(x1,hidden,w1,w2,b1,b2,output,0)
        result1 = o5
        print_all(x1, 0, h, output, o5, e, e3, e4, w1, w2, b1, b2)

        #0 and 1
        h = b1 + np.dot(x2, w1)
        hidden = activate_hidden(h)
        output = b2 + np.dot(hidden,w2)
        w1,w2,b1,b2,o5,e,e3,e4 = shift_network(x2,hidden,w1,w2,b1,b2,output,1)
        result2 = o5
        print_all(x2, 1, h, output, o5, e, e3, e4, w1, w2, b1, b2)

        #1 and 0 
        h = b1 + np.dot(x3, w1)
        hidden = activate_hidden(h)
        output = b2 + np.dot(hidden,w2)
        w1,w2,b1,b2,o5,e,e3,e4 = shift_network(x3,hidden,w1,w2,b1,b2,output,1)
        result3 = o5
        print_all(x3, 1, h, output, o5, e, e3, e4, w1, w2, b1, b2)


        #1 and 1 
        h = b1 + np.dot(x4, w1)
        hidden = activate_hidden(h)
        output = b2 + np.dot(hidden,w2)
        w1,w2,b1,b2,o5,e,e3,e4 = shift_network(x4,hidden,w1,w2,b1,b2,output,0)
        result4 = o5
        print_all(x4, 0, h, output, o5, e, e3, e4, w1, w2, b1, b2)

        i = i + 1
    
    if(result1 == 0 and result2 == 1 and result3 == 1 and result4 == 0):
        break

    
print(w1)
print(w2)