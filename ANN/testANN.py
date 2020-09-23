import numpy as np


def feed(N,x,y):
    count=0
    for i in range(N):
    a=[]
    X=np.reshape(x[i],(784,1))
    Z1=nn.weight[0].dot(X)+nn.bias[0]
    A1=nn.RELU(Z1)
    Z2=nn.weight[1].dot(A1)+nn.bias[1]
    out=nn.softmax(Z2)
    predict=np.argmax(out,axis=0)
    real=np.argmax(np.reshape(y[i],(10,1)),axis=0)
    if predict[0]==real[0]:
        count=count+1
    count=count*100/N

    return count 