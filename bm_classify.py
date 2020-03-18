import numpy as np


def binary_train(X, y, loss="perceptron", w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: binary training labels, a N dimensional numpy array where 
    N is the number of training points, indicating the labels of 
    training data
    - loss: loss type, either perceptron or logistic
    - step_size: step size (learning rate)
	- max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of logistic or perceptron regression
    - b: scalar, which is the bias of logistic or perceptron regression
    """
    N, D = X.shape
    assert len(np.unique(y)) == 2


    w = np.zeros(D)
    if w0 is not None:
        w = w0
    
    b = 0
    if b0 is not None:
        b = b0
     
    y_new=np.where(y==0,-1,1)

    if loss == "perceptron":
        ############################################
        # TODO 1 : Edit this if part               #
        #          Compute w and b here            #
        w = np.zeros(D)
        b = 0
        # Xlist=X.tolist()
        # p=len(Xlist)
        # y1=y
        # for i in range(len(y1)):
        #     if y[i]==0:
        #         y1[i]=-1
        # one=np.ones((p,1))
        # oneW=np.concatenate((one,X),axis=1)
        # s=np.insert(w,0,b)
        for i in range(max_iterations):
        #     #print(s.shape,oneW.shape,y1.shape)
        #     t=np.dot(s,oneW.transpose())
              t=X.dot(w)+b
              t=y_new*t
              #error = []
              #for i in t:
                 # if t<=0:
                  #    error.append(1.0)
                 # else:
                 #     error.append(0.0)
              #error_adjusted = np.array(error)  
              error=(t<=0).astype(float)
              error=y_new*error
                
        #     for i in range(len(t)):
        #         if t[i]>0 and y1[i]>0:
        #             t[i]=0
        #         else:
        #             t[i]=1
              w=w + step_size*(np.dot(error,X))/N
              b=b + step_size*(np.sum(error))/N
        #     s=s+u
        # #print(t)
        # b=s[0]
        # w=np.delete(s,0,axis=0)
        #print(w,b)
        ############################################
        

    elif loss == "logistic":
        ############################################
        # TODO 2 : Edit this if part               #
        #          Compute w and b here            #
        
        Xlist=X.tolist()
        p=len(Xlist)
        one=np.ones((p,1))
        oneW=np.concatenate((one,X),axis=1)
        s=np.insert(w,0,b)
        for _ in range(max_iterations):
            t=sigmoid(np.dot(s,oneW.transpose()))-y
            u1=step_size*np.dot(t,oneW)
            u=u1/N
            s=s-u
        b=s[0]
        #print(b)
        #w=np.delete(s,0,axis=0)
        w = s[1:]
        #print(w)
        #w = np.zeros(D)
        #b = 0
        ############################################
        

    else:
        raise "Loss Function is undefined."

    assert w.shape == (D,)
    return w, b

def sigmoid(z):
    
    """
    Inputs:
    - z: a numpy array or a float number
    
    Returns:
    - value: a numpy array or a float number after computing sigmoid function value = 1/(1+exp(-z)).
    """

    ############################################
    # TODO 3 : Edit this part to               #
    #          Compute value                   #
    value = 1 / (1 + np.exp(-z))
    ############################################
    
    return value

def binary_predict(X, w, b, loss="perceptron"):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of your learned model
    - b: scalar, which is the bias of your model
    - loss: loss type, either perceptron or logistic
    
    Returns:
    - preds: N dimensional vector of binary predictions: {0, 1}
    """
    N, D = X.shape
    
    if loss == "perceptron":
        ############################################
        # TODO 4 : Edit this if part               #
        #          Compute preds                   #
        preds = np.zeros(N)
        # Xlist=X.tolist()
        # p=len(Xlist)
        # one=np.ones((p,1))
        # oneW=np.concatenate((one,X),axis=1)
        #preds=np.zeros(N)
        Z=np.dot(w,X.T)+b
        # s=np.insert(w,0,b)
        for i in range(N):
        #     #print(s.shape,oneW.shape,y1.shape)  
        # preds=np.dot(s,oneW.transpose())
        # preds = np.array([1 if i >= 0.5 else 0 for i in preds])
        #assert preds.shape == (N,) 
            #for i in range(len(preds)):
                if Z[i]<0:
                    preds[i]=0
                else:
                    preds[i]=1
            #assert preds.shape == (N,)        
        ############################################
        

    elif loss == "logistic":
        ############################################
        # TODO 5 : Edit this if part               #
        #          Compute preds                   #
        preds = np.zeros(N)
        preds = sigmoid(w.dot(X.T) + b)
        L=[]
        for i in preds:
            if i >= 0.5:
                L.append(1)
            else:
                L.append(0)
        preds= np.array(L)       
        #preds = np.array([1 if i >= 0.5 else 0 for i in preds])
        #print(preds)
        #assert preds.shape == (N,) 
        ############################################
        

    else:
        raise "Loss Function is undefined."
    

    assert preds.shape == (N,) 
    return preds



def multiclass_train(X, y, C,
                     w0=None, 
                     b0=None,
                     gd_type="sgd",
                     step_size=0.5, 
                     max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of 
    training data
    - C: number of classes in the data
    - gd_type: gradient descent type, either GD or SGD
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: C-by-D weight matrix of multinomial logistic regression, where 
    C is the number of classes and D is the dimensionality of features.
    - b: bias vector of length C, where C is the number of classes
    """

    N, D = X.shape

    w = np.zeros((C, D))
    if w0 is not None:
        w = w0
    
    b = np.zeros(C)
    if b0 is not None:
        b = b0
        
    np.random.seed(42)
    if gd_type == "sgd":
        ############################################
        # TODO 6 : Edit this if part               #
        #          Compute w and b                 #
        w = np.zeros((C, D))
        b = np.zeros(C)
        #y = np.eye(C)[y]
        Xlist=X.tolist()
        p=len(Xlist)
        one=np.ones((p,1))
        oneW=np.concatenate((X,one),axis=1)
        s=np.insert(w,w.shape[1],b,axis=1)
        #print(N,X.shape,y.shape)
        #print("s = ",s)
        for i in range(max_iterations):
            k=np.random.randint(0,N)
            
            error = np.matrix(softmax((oneW[k].dot(s.T)))).T
            #print (error.shape)
            #print(y[k])
            error[y[k]]=error[y[k]]-1
            
            #print(C,D,error.shape)
            #print(error[k],"e1")
            #error[k]-=y[k]
            #print(error[k],y[k],"e2")
            #print(error.shape)
            #for j in range(0,len(error[k])):
            #    error[k][j]-=1
            #print(error.shape,y.shape)
            #np.reshape(error,(C,1))
            #print(error.T.shape,np.matrix(oneW[k]).shape)
            #print(np.matrix(oneW[k]).shape)
            #exit()
            w_gradnt = np.dot(error,np.matrix(oneW[k]))
            #print(w_gradient)
            #break
            #b_gradient = np.sum(error, axis=0) / N
            s -= step_size * w_gradnt
            #b -= step_size * b_gradient
        b=s[:,len(s[0])-1]
        w = np.delete(s, len(s[0])-1, 1) 
        
        #print(s)
        #print(w,b)
        #exit()
        ############################################
        

    elif gd_type == "gd":
        ############################################
        # TODO 7 : Edit this if part               #
        #          Compute w and b                 #
        w = np.zeros((C, D))
        b = np.zeros(C)
        y = np.eye(C)[y]
        for i in range(max_iterations):
            error = softmax1((w.dot(X.T)).T + b) - y
            w_gradnt1 = error.T.dot(X) 
            w_gradnt = w_gradnt1/N
            #print(w_gradient)
            b_gradnt1 = np.sum(error, axis=0)
            b_gradnt = b_gradnt1/N
            w -= step_size * w_gradnt
            b -= step_size * b_gradnt
        ############################################
        

    else:
        raise "Type of Gradient Descent is undefined."
    

    assert w.shape == (C, D)
    assert b.shape == (C,)

    return w, b


def softmax1(X):
    X = np.exp(X - np.amax(X))
    denom = np.sum(X,axis=1)
    return (X.T / denom).T

def multiclass_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: weights of the trained multinomial classifier, C-by-D 
    - b: bias terms of the trained multinomial classifier, length of C
    
    Returns:
    - preds: N dimensional vector of multiclass predictions.
    Outputted predictions should be from {0, C - 1}, where
    C is the number of classes
    """
    N, D = X.shape
    ############################################
    # TODO 8 : Edit this part to               #
    #          Compute preds                   #
    #preds = np.zeros(N)
    
    preds = np.zeros(N)
    #return preds
    preds = softmax((w.dot(X.T)).T + b)
    preds = np.argmax(preds, axis=1)

    #def train(self, X,gd_type, y):
        #if gd_type == "sgd":
           
        #w = np.zeros((C, D))
        #b = np.zeros(C)
        #y = np.eye(C)[y]
        #Xlist=X.tolist()
       # p=len(Xlist)
        #one=np.ones((p,1))
        #oneW=np.concatenate((X,one),axis=1)
       # s=np.concatenate((w,b),axis=1)
        #print(N,X.shape,y.shape)
        
        
            #print(error.shape)
        

    
        
    ############################################

    assert preds.shape == (N,)
    return preds

def softmax(X):
    X = np.exp(X - np.amax(X))
    denom = np.sum(X)
    return (X.T / denom).T



    