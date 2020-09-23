import numpy as np
import math
## class neural network
class neu_net:
    def __init__(self,layers,alpha=0.0001,gamma=0.9):
        self.layers=layers
        self.weight=[]## list, chứa L-1 phần tử,
        self.bias=[]## list, chứa L-1 phần tử
        self.v_weight=[] ## list, chứa L-1 phần tử,
        self.v_bias=[]## list, chứa L-1 phần tử,
        self.alpha=alpha
        self.gamma=gamma
        self.L=len(self.layers)
        ## create neuron network: weight and bias
        for i in range(self.L-1):
                w,b=self.init_tech(self.layers[i],self.layers[i+1])
                v1=np.zeros((self.layers[i+1],self.layers[i]))
                ##verlocty ban đầu bằng o
                self.v_weight.append(v1)
                v2=np.zeros((self.layers[i+1],1))
                self.v_bias.append(v2)
                self.weight.append(w)
                self.bias.append(b)

    def init_tech(self,fan_in, fan_out):
        init_bound= np.sqrt(6/(fan_in+fan_out))
        w=np.random.uniform(-init_bound,init_bound,(fan_out,fan_in))
        b=np.random.uniform(-init_bound, init_bound,(fan_out,1))
        return w, b

    def softmax(self,z):## active in last layer với tổng xác xuất ngỏ ra bằng 1 để có thể áp dụng được cross entropy   
        n=max(z)
        k=np.exp(z-n)## tránh bị overflow khi tính toán
        temp=k/np.sum(k)
        return temp

    def RELU(self,z):#active in hidden layers
        for i in range(len(z)):
        z[i]=max(0.001*z[i],z[i])## leaky relu
        return z

    def loss(self, y_pre,y):
        temp=-np.sum((y*np.log(y_pre)))
        return temp
    
    def feed_fw(self,a):
        for i in range(self.L-2): ## layer cuoi dug softmax
                z=self.weight[i].dot(a[-1])+self.bias[i]
                ## pass throught activation function RELU
                temp=self.RELU(z)
                a.append(temp)
        z=self.weight[-1].dot(a[-1])+self.bias[-1]
        y=self.softmax(z)
        a.append(y)
        return a # a list, chứa L phan tử, 0th chứa input

    def back_pro(self,a,y):
        weight_gd=[]
        bias_gd=[]
        ## dao ham tai layer cuoi
        delta_L=(a[-1]-y)
        bias_gd.append(delta_L)
        weight_gd.append(delta_L.dot(a[-2].T))
        ## tinh cho cac layer con lai
        temp=delta_L
        for i in range(self.L-2,0,-1):
                temp=self.weight[i].T.dot(temp)
                temp[a[i]<=0]=0.001
                bias_gd.append(temp)
                weight_gd.append(temp.dot(a[i-1].T)) 
        weight_gd.reverse()
        bias_gd.reverse()
        return weight_gd,bias_gd ## giá trị đạo hàm của hàm loss theo  weigt và bias
    
       
