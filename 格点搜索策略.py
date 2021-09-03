import numpy as np
from sklearn import svm
from sklearn.model_selection import LeaveOneOut
data = np.eye(20)
amino_acid = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
file = open('三肽序列.txt','r')
line = file.read()
line_1 = line.replace('\n','')
file_2 = open('三肽序列.txt','r')
line_2 = file_2.readlines()
number = len(line_2)
data_1 = np.zeros((number,20))
non = np.zeros((number*3, 20))
a = 0
for i in range(number*3):
    if line_1[i] in amino_acid:  
        non[a][amino_acid.index(line_1[i])] = 1
        a += 1        
X = non.reshape(number,60)
y = np.loadtxt('活性值.txt')
data = np.zeros((number,20))
for i in range(number):
    a = line_1[i*3:(i+1)*3]   
    result = {}
    for b in a:        
        result[b] = result.get(b,0) + 1              
    allsum = sum(result.values())  
    math = list(result.items())
    if len(math) == 3:          
        x = 0
        data_1[i][amino_acid.index(math[x][0])] = math[x][1]/allsum        
        x = 1
        data_1[i][amino_acid.index(math[x][0])] = math[x][1]/allsum        
        x = 2
        data_1[i][amino_acid.index(math[x][0])] = math[x][1]/allsum        
    if len(math) == 2:
        x = 0
        data_1[i][amino_acid.index(math[x][0])] = math[x][1]/allsum        
        x = 1
        data_1[i][amino_acid.index(math[x][0])] = math[x][1]/allsum
data_2 = np.zeros((number,400))
all_data = amino_acid*20
for i in range(20):        
    all_data[i*20+0] += amino_acid[i]
    all_data[i*20+1] += amino_acid[i]
    all_data[i*20+2] += amino_acid[i]
    all_data[i*20+3] += amino_acid[i]
    all_data[i*20+4] += amino_acid[i]
    all_data[i*20+5] += amino_acid[i]
    all_data[i*20+6] += amino_acid[i]
    all_data[i*20+7] += amino_acid[i]
    all_data[i*20+8] += amino_acid[i]
    all_data[i*20+9] += amino_acid[i]
    all_data[i*20+10] += amino_acid[i]
    all_data[i*20+11] += amino_acid[i]
    all_data[i*20+12] += amino_acid[i]
    all_data[i*20+13] += amino_acid[i]
    all_data[i*20+14] += amino_acid[i]
    all_data[i*20+15] += amino_acid[i]
    all_data[i*20+16] += amino_acid[i]
    all_data[i*20+17] += amino_acid[i]
    all_data[i*20+18] += amino_acid[i]
    all_data[i*20+19] += amino_acid[i]
for i in range(number):
    if line[i*4:i*4+2] in all_data:
        data_2[i,all_data.index(line[i*4:i*4+2])] = 1       
    if line[i*4+1:i*4+3] in all_data:
        data_2[i,all_data.index(line[i*4+1:i*4+3])] = 1  
X = np.c_[X,data_1,data_2]
y_mean = np.full((1,number), np.sum(y)/number)
loo = LeaveOneOut()
loo.get_n_splits(X)
count = -1
Q2 = np.zeros((576,4))
for a in range(-1,7):
    c = 2**a 
    for i in range(-8,0):
        p = 2**i
        for b in range(-8,1):
            g = 2**b 
            YPred = np.zeros(number)        
            count = count + 1
            for train_index, test_index in loo.split(X):    
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]             
                regr = svm.SVR(kernel ='rbf',degree = 3,gamma = g ,coef0 = 0.0,
		    tol = 0.001,C = c,epsilon = p,shrinking = True,cache_size = 40,		
		    verbose = False,max_iter = -1)            
                regr.fit(X_train, y_train)            
                y_pre = regr.predict(X_test)
                YPred[test_index] = y_pre         
            num = (1-np.sum((y-YPred)**2)/np.sum((y-y_mean)**2))
            Q2[count][0] = c
            Q2[count][1] = p
            Q2[count][2] = g
            Q2[count][3] = num            
new = Q2[np.argsort(Q2[:,3])]#在Q2结果的最后一行依次是C，epsilon，gamma优化值，最后一列为交叉验证结果。 
print('优化后最高Q2：',num)
print('最优C值：{}，最优epsilon值：{}，最优gamma值：{}'.format(c,p,g))
file.close()
file_2.close()           

              

