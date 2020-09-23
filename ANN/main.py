import numpy as np
from readdata  import *
from neuronNet import *
from testANN import *
import pickle as pk
layer=[784,512,10]
nn=neu_net(layer)
L=len(layer)
N_train=55000##train-images
N_vali=5000
N_test=10000
d=x.shape[1]
d1=10## numbers of class
a=[]
epoch =21
## sgd
cost=0
acc_train=[]
acc_vali=[]
acc_test=[]
loss=[]
for i in range (epoch):
 ## shuffle data để đảm bảo tính random của dữ liệu
    cost=0
    id_x=np.random.permutation(N_train) ## xáo trộn vi trí của phần tử mảng
    for k in id_x:
        a=[]
        s=np.reshape(x_train[k],(d,1))
        a.append(s)
        a=nn.feed_fw(a)
        y_hat=np.reshape(y_train[k],(d1,1))
        cost=cost-np.sum(y_hat*np.log(a[-1]))
        w,b=nn.back_pro(a,y_hat)
        for m in range(2):##using momentum for update
                nn.v_weight[m]=nn.gamma*nn.v_weight[m]+nn.alpha*w[m]
                nn.v_bias[m]=nn.gamma*nn.v_bias[m]+nn.alpha*b[m]
                nn.weight[m]=nn.weight[m]-nn.v_weight[m]
                nn.bias[m]=nn.bias[m]-nn.v_bias[m]
        if i%2==0:
            print(a[-1])
            loss.append(cost/N_train)
            print("iter:",i,"loss: %3f" %(cost/N_train))
            ##tinh train_acc
            acc_train.append(feed(N_train,x_train,y_train))
            acc_vali.append(feed(N_vali,x_vali,y_vali))
            acc_test.append(feed(N_test,x_test,y_test))

pk.dump( nn.weight , open( 'weight_do_an.pkl' , 'wb' ) )
pk.dump( nn.bias, open( 'bias_do_an.pkl' , 'wb' ) )

iters=[i for i in range(21) if i%2==0]
## ve do thi
import matplotlib.pyplot as plt
plt.plot(iters,acc_train,color='red',label="train accuracy")
plt.plot(iters,acc_vali,color='green',label="validation accuracy")
plt.plot(iters,acc_test,color='blue',label="test accuracy")
plt.legend(loc="best")
plt.grid()
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.show()