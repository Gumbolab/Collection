##readinng file train
import pandas as pd
import time
start= time.time()
train=pd.read_csv("M:/data/mnist_train.csv")
train=np.array(train)
train=train.T
label_train=train[0]
x=train[1:].T
##one-hot ecoding train label
Y_hat=np.zeros((label_train.shape[0],10))
Y_hat[np.arange(label_train.shape[0]),label_train]=1
## tách tập train thành train set và validation set
x_train=x[0:55000]/255
y_train=Y_hat[0:55000]
x_vali=x[55000:60000]/255 ## để dễ tính toán
y_vali=Y_hat[55000:60000]
print(x_train.shape)#(shape=(55.000,784)
print(y_train.shape) #shape=(5000,10)
end=time.time()
print(end-start)
##reading file test
read_test=pd.read_csv("M:/data/mnist_test.csv")
##ma tran ngo vao va ngo ra file test
read_test=np.array(read_test).T
label_test=read_test[0]
x_test=read_test[1:].T
x_test=x_test/255
##one-hot encoding
y_test=np.zeros((label_test.shape[0],10))
y_test[np.arange(label_test.shape[0]),label_test]=1
print(x_test.shape)
print(y_test.shape)
