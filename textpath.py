import numpy as np
import matplotlib.pyplot as plt

def Too_sharp_bend(data_input,arc_two):

          end_begin_dis=np.sqrt(np.sum(np.square(data_point[0]-data_point[-1])))
          total_length=np.sum(arc_two)
          if total_length>=3*end_begin_dis:
              return True
          else:
            return False

def remove_duplicated_point(old_data_point):
          data_point=[]
          L=len(old_data_point)
          temp=old_data_point[0]
          data_point.append(temp)
          #remove all consecutive point have the same coordinate
          for i in range(L):
                if (old_data_point[i] != temp).any():
                    data_point.append(old_data_point[i])
                    temp=old_data_point[i]
          data_point=np.array(data_point)
          return data_point

def normalize_coodinate(datapoint):       
          datapoint.T[0]=(datapoint.T[0]-np.min(datapoint.T[0]))/(np.max(datapoint.T[0])-np.min(datapoint.T[0]))
          datapoint.T[1]=datapoint.T[1]-np.min(datapoint.T[1])
          datapoint.T[1]=datapoint.T[1]/(np.max(datapoint.T[1])-np.min(datapoint.T[1]))
          
def compute_angle(old_data_point):           
          L=len(data_point)
          
          #for i in range(L-1):
          arc_two=np.sqrt(np.sum(np.square(data_point[0:L-1]-data_point[1:L]),axis=1))

          #for j in range(L-2):
          arc_three=np.sqrt(np.sum(np.square(data_point[0:L-2]-data_point[2:L]),axis=1))
          
          #index i in ratio angle= i+1 in data_point
         
          ratio_angle=np.array([(arc_two[i]+arc_two[i+1])/arc_three[i] for i in range(arc_three.shape[0])]    )               
          return ratio_angle,arc_two

def split_too_sharp(data_point,ratio_angle,arc_two):

          index_of_split_point=np.argmax(ratio_angle[2:-2])+2+1

          data_point_sub_1=[data_point[:index_of_split_point+1],
                            ratio_angle[:index_of_split_point-1],arc_two[:index_of_split_point]]
          data_point_sub_2=[data_point[index_of_split_point:]
                           , ratio_angle[index_of_split_point+1:],arc_two[index_of_split_point:]]

          return data_point_sub_1,data_point_sub_2 #,index_of_split_point


def split_B_loss(data_point,ratio_angle,arc_two):

          index_of_split_point=np.argmin(ratio_angle[2:-2])+2+1

          data_point_sub_1=[data_point[:index_of_split_point+1],
                            ratio_angle[:index_of_split_point-1],arc_two[:index_of_split_point]]
          data_point_sub_2=[data_point[index_of_split_point:]
                           , ratio_angle[index_of_split_point+1:],arc_two[index_of_split_point:]]

          return data_point_sub_1,data_point_sub_2 #,index_of_split_point
#compute first and second order deviration of (4) equation,x'(si)(xi − x(si)) + y'(si)(yi − y(si)) = 0,
def order_value(S,XY,cof):
          """
          input S,XY,XY_si,cof
          return value of first order different,vector of 1-dimension vector
          """
          si=S.T[1]
          S_first=np.array([[0,1,2*t,3*t**2] for t in si])
          S_second=np.array([[0,0,2,6*t] for t in si])
          XY_si=S.dot(cof)
          del_xy=XY-XY_si
          term_1=np.sum(S_second.dot(cof)*del_xy,axis=1)
          term_2=np.sum(np.square(S_first.dot(cof)),axis=1)
          second=term_1-term_2
          first=np.sum(S_first.dot(cof)*del_xy,axis=1)
          return first,second

#update s
def Newton_method(S,XY,cof):
          """
          input is XY,XY_si,S,cof
          return value of second-order defferent,scalar
          update s in bezier curve
          """
          first_order,second_order=order_value(S,XY,cof)
          #update si
          si=S.T[1]
          #update si
          si=si-first_order/second_order
          S=np.array([[1,t,t**2,t**3] for t in si])
          return S
def compute_loss(xyt,xyt_Bz):   
          return np.mean(np.square(xyt-xyt_Bz))

def bezier_cof(data_point,S):
          cof=np.linalg.inv(S.T.dot(S)).dot(S.T).dot(data_point)
          return cof 

def Find_BezierCurve(data_point):
          # data_point is an array         
          N=data_point.shape[0]
          
          epoch=10
          #init matrix s voi cac gia tri cach deu nhau voi shape banng voi so datapoint
          s_init=np.arange(0,N,1)/N
          S=np.array([[1,t,t**2,t**3] for t in s_init]).astype('f4')
          loss=[]
          for i in range(epoch):
                S[0]=np.array([1,0,0,0])
                S[-1]=np.array([1,1,1,1])
                cof=bezier_cof(data_point,S)
              
                for _ in range(50):
                      S= Newton_method(S,data_point,cof)
                xy_bz=S.dot(cof)
                loss.append(compute_loss(data_point,xy_bz))
    
          
          return loss,cof,S.dot(cof)


# tao datapoint
data = """857 853 ,825 799 ,825 799 ,825 799 ,825 799
                 ,825 799 ,825 799 ,795 743 ,795 743 ,795 743
                 ,795 743 ,795 743 ,795 743 ,830 685 ,830 685
                 ,830 685 ,830 685 ,888 657 ,888 657 ,888 657
                 ,942 655 ,942 655 ,942 655 ,1005 671 ,1005 671
                 ,1005 671 ,1059 705 ,1059 705 ,1059 705 ,1059 705
                 ,1103 761 ,1103 761 ,1103 761 ,1103 761 ,1098 819
                 ,1098 819 ,1098 819 ,1098 819 ,1098 819 ,1040 869
                 ,1040 869 ,1040 869 ,1040 869 ,1040 869 ,980 888
                 ,980 888 ,980 888 ,980 888 ,980 888 ,980 888 ,980 888
                 ,980 888 ,980 888 ,980 888 ,1031 887 ,1031 887 ,1031 887
                 ,1084 897 ,1084 897 ,1084 897 ,1140 920 ,1140 920 ,1140 920
                 ,1194 957 ,1194 957 ,1194 957 ,1194 957 ,1238 1024 ,1238 1024
                 ,1238 1024 ,1240 1083 ,1240 1083 ,1240 1083 ,1205 1139 ,1205 1139
                 ,1205 1139 ,1142 1180 ,1142 1180 ,1142 1180 ,1070 1209 ,1070 1209
                 ,1070 1209 ,1007 1213 ,1007 1213 ,1007 1213""".split(",")
data = [list(map(int,x.strip().split(" "))) for x in data]
data_input = np.array(data).astype('f4')

# remove duplicate point
data_point=remove_duplicated_point(data_input)
#normalize coordinate
normalize_coodinate(data_point)

plt.plot(data_point.T[0],data_point.T[1],'ro')


#prepare for find BZ
# compute khoang cach va ti le khoang cah 3/2
ratio,arc_two=compute_angle(data_point)
def SP_cur(data_point,ratio,arc_two):
          sub_data=[]
          sub_cof=[]
          if Too_sharp_bend(data_point,arc_two) and len(data_point)>=7:# split in to two curve
                    d1,d2=split_too_sharp(data_point,ratio,arc_two)#[data,ratio,arc_two]
                    #print(np.array(d1))
                    d,c=SP_cur(d1[0],d1[1],d1[2])                   
                    sub_data+=d#doan data
                    sub_cof+=c#bezier  
                    d,c=SP_cur(d2[0],d2[1],d2[2])
                    sub_data+=d
                    sub_cof+=c
          else:       
 
                      loss,cof,Bezier=Find_BezierCurve(data_point)    
                      plt.plot(Bezier.T[0],Bezier.T[1])
                      
                    
                      if loss[-1]>=0.00001 and len(data_point)>=7:
                              d1,d2=split_B_loss(data_point,ratio,arc_two)#[data,ratio,arc_two]
                              #print(np.array(d1))
                              d,c=SP_cur(d1[0],d1[1],d1[2])                   
                              sub_data+=d
                              sub_cof+=c
                              d,c=SP_cur(d2[0],d2[1],d2[2])
                              sub_data+=d
                              sub_cof+=c

                      else:
                     
                            sub_data.append([data_point,ratio,arc_two])#[data,ratio,arc_two]
                            sub_cof.append(cof)
                            
          return sub_data,sub_cof
         




sub,cof=SP_cur(data_point,ratio,arc_two)
print(len(sub))
