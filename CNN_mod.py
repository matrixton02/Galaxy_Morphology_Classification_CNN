import numpy as np
import mlp as mlp
import data_load as dl
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

#thsi convers the image in column format and then we perform matrix multiplication once instead of repeated kernal overallping and multiplications
def im2col(X,kernel_size,stride,padding):
    N,C,H,W=X.shape

    if(padding>0):
        X=np.pad(X,((0,0),(0,0),(padding,padding),(padding,padding)))

    H_p,W_p=X.shape[2],X.shape[3]

    H_out=(H_p-kernel_size)//stride+1
    W_out=(W_p-kernel_size)//stride+1

    cols=np.zeros((N,C*kernel_size*kernel_size,H_out*W_out))

    for i in range(H_out):
        for j in range(W_out):
            h_start=i*stride
            w_start=j*stride

            patch=X[:,:,h_start:h_start+kernel_size,w_start:w_start+kernel_size]

            cols[:,:,i*W_out+j]=patch.reshape(N,-1)

    return cols,H_out,W_out

#in converst the image back from column to image format
def col2im(cols,X_shape,kernel_size,stride,padding):
    N,C,H,W=X_shape

    H_p=H+2*padding
    W_p=W+2*padding

    X_padded=np.zeros((N,C,H_p,W_p))

    H_out=(H_p-kernel_size)//stride+1
    W_out=(W_p-kernel_size)//stride+1

    for i in range(H_out):
        for j in range(W_out):
            h_start=i*stride
            w_start=j*stride

            patch=cols[:,:,i*W_out+j]
            patch=patch.reshape(N,C,kernel_size,kernel_size)

            X_padded[:,:,h_start:h_start+kernel_size,w_start:w_start+kernel_size]+=patch

    if padding>0:
        return X_padded[:,:,padding:-padding,padding:-padding]
    
    return X_padded

# the relu fuction class 
class ReLU:
    def forward(self,X):
        self.X=X
        return np.maximum(0,X)

    def backward(self,dY):
        return dY*(self.X>0)
    
class MaxPool:
    def __init__(self,kernel_size=2,stride=2):
        self.kernel_size=kernel_size
        self.stride=stride

    def forward(self,X):
        self.X=X
        N,C,H,W=X.shape

        H_out=(H-self.kernel_size)//self.stride+1
        W_out=(W-self.kernel_size)//self.stride+1

        Y=np.zeros((N,C,H_out,W_out))
        self.cache={}

        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start=i*self.stride
                        w_start=j*self.stride

                        patch=X[n,c,h_start:h_start+self.kernel_size,w_start:w_start+self.kernel_size]
                        Y[n,c,i,j]=np.max(patch)

                        self.cache[(n,c,i,j)]=patch
        return Y
    
    def backward(self,dY):
        N,C,H_out,W_out=dY.shape
        dX=np.zeros_like(self.X)

        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start=i*self.stride
                        w_start=j*self.stride

                        patch=self.X[n,c,h_start:h_start+self.kernel_size,w_start:w_start+self.kernel_size]
                        max_val=np.max(patch)

                        for m in range(self.kernel_size):
                            for k in range(self.kernel_size):
                                if patch[m][k]==max_val:
                                    dX[n,c,h_start+m,w_start+k]+=dY[n,c,i,j]
        return dX
    
class Flatten:
    def forward(self,X):
        self.shape=X.shape
        return X.reshape(X.shape[0],-1)
    
    def backward(self,dY):
        return dY.reshape(self.shape)
    
class Conv2d:
    def __init__(self,C_in,C_out,kernel_size,stride=1,padding=0):
        self.C_in=C_in
        self.C_out=C_out
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding=padding

        self.weights=np.random.randn(C_out,C_in,kernel_size,kernel_size)*np.sqrt(2/(C_in*kernel_size*kernel_size))
        self.bias=np.zeros(C_out)

    def forward(self,X):
        """X shape is (N,C_in,H,W) Y shape is (N,C_out,H_out,W_out)"""
        self.X=X
        cols,H_out,W_out=im2col(X,self.kernel_size,self.stride,self.padding)

        self.cols=cols

        W_col=self.weights.reshape(self.C_out,-1)

        Y=np.zeros((X.shape[0],self.C_out,H_out*W_out))

        for n in range(X.shape[0]):
           Y[n]=W_col@cols[n]+self.bias[:,None]

        Y = Y.reshape(X.shape[0], self.C_out, H_out, W_out)
        return Y
    
    def backward(self,dY):
        """dY shape is (N,C_out,H_out,W_out)"""
        N=dY.shape[0]

        dY_col=dY.reshape(N,self.C_out,-1)
        W_col=self.weights.reshape(self.C_out,-1)

        dW=np.zeros_like(W_col)
        dcols=np.zeros_like(self.cols)

        for n in range(N):
            dW+=dY_col[n]@self.cols[n].T
            dcols[n]=W_col.T@dY_col[n]
        
        self.dW=dW.reshape(self.weights.shape)/N
        self.db=np.sum(dY,axis=(0,2,3))/N

        dX=col2im(dcols,self.X.shape,self.kernel_size,self.stride,self.padding)
        return dX
    
class CNN:
    def __init__(self):
        self.conv1=Conv2d(3,16,3,stride=1,padding=1)
        self.relu1=ReLU()
        self.pool1=MaxPool(2,2)

        self.conv2=Conv2d(16,32,3,stride=1,padding=1)
        self.relu2=ReLU()
        self.pool2=MaxPool(2,2)

        self.conv3=Conv2d(32,64,3,stride=1,padding=1)
        self.relu3=ReLU()
        self.pool3=MaxPool(2,2)

        self.flatten=Flatten()
    
    def forward(self, X):
        X = self.conv1.forward(X)
        X = self.relu1.forward(X)
        X = self.pool1.forward(X)

        X = self.conv2.forward(X)
        X = self.relu2.forward(X)
        X = self.pool2.forward(X)

        X = self.conv3.forward(X)
        X = self.relu3.forward(X)
        X = self.pool3.forward(X)

        X = self.flatten.forward(X)
        return X
    
    def backward(self, dY):
        dY = self.flatten.backward(dY)

        dY = self.pool3.backward(dY)
        dY = self.relu3.backward(dY)
        dY = self.conv3.backward(dY)

        dY = self.pool2.backward(dY)
        dY = self.relu2.backward(dY)
        dY = self.conv2.backward(dY)

        dY = self.pool1.backward(dY)
        dY = self.relu1.backward(dY)
        dY = self.conv1.backward(dY)

        return dY
    
def cross_entropy_loss(y_pred, y_true):
    m = y_true.shape[1]
    loss = -(1/m)*np.sum(y_true*np.log(y_pred+1e-9))
    dZ = (y_pred - y_true)/m
    return loss, dZ

def train(cnn,X,Y,mlp_parameters,config,learning_rate=0.001,epochs=10,batch_size=32):
    N=X.shape[0]
    cost_list=[]
    print("Training Started....\n")
    for epoch in range(epochs):
        indices=np.random.permutation(N)
        X_shuffled=X[indices]
        Y_shuffled=Y[:,indices]

        epoch_loss=0
        num_batches=0

        for i in range(0,N,batch_size):
            X_batch=X_shuffled[i:i+batch_size]
            Y_batch=Y_shuffled[:,i:i+batch_size]

            features=cnn.forward(X_batch)
            features_t=features.T

            AL,cache=mlp.forward_propagation(features_t,mlp_parameters,config)

            loss=mlp.cost_fucntion(AL,Y_batch,config)
            epoch_loss+=loss
            num_batches+=1

            gradients,dA_input=mlp.back_propagation(features_t,Y_batch,mlp_parameters,cache,config)

            d_flat=dA_input.T
            cnn.backward(d_flat)

            mlp_parameters=mlp.update_parameters(mlp_parameters,gradients,learning_rate)

            cnn.conv1.weights-=learning_rate*cnn.conv1.dW
            cnn.conv1.bias-=learning_rate*cnn.conv1.db

            cnn.conv2.weights-=learning_rate*cnn.conv2.dW
            cnn.conv2.bias-=learning_rate*cnn.conv2.db
        
        avg_loss=epoch_loss/num_batches
        cost_list.append(avg_loss)

        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.5f}")

    return mlp_parameters,cost_list

def compute_accuracy(cnn, mlp_parameters, config, X, Y):
    features = cnn.forward(X)
    features_t = features.T

    AL, _ = mlp.forward_propagation(features_t,mlp_parameters,config)
    predictions = np.argmax(AL, axis=0)
    true_labels = np.argmax(Y, axis=0)

    accuracy = np.mean(predictions == true_labels) * 100

    return accuracy
def plot_confusion_matrix(cnn, mlp_parameters, config, X, Y, class_names):

    features = cnn.forward(X)
    features_t = features.T

    AL, _ = mlp.forward_propagation(
        features_t,
        mlp_parameters,
        config
    )

    preds = np.argmax(AL, axis=0)
    true = np.argmax(Y, axis=0)

    cm = confusion_matrix(true, preds)

    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

def save_model(cnn, mlp_parameters, filepath):
    model_data = {
        "cnn": cnn,
        "mlp_parameters": mlp_parameters
    }

    with open(filepath, "wb") as f:
        pickle.dump(model_data, f)

def load_model(filepath):
    with open(filepath, "rb") as f:
        model_data = pickle.load(f)
    return model_data["cnn"], model_data["mlp_parameters"]

def show_prediction(cnn,mlp_parameters,config,X,Y,class_names,num_samples=5):
    indices=np.random.choice(X.shape[0],num_samples,replace=False)

    for idx in indices:
        img=X[idx:idx+1]
        features=cnn.forward(img)
        features_t=features.T

        AL,_=mlp.forward_propagation(features_t,mlp_parameters,config)
        pred = np.argmax(AL, axis=0)[0]
        true = np.argmax(Y[:, idx])

        plt.imshow(img[0].transpose(1,2,0))
        plt.title(f"Pred: {class_names[pred]} | True: {class_names[true]}")
        plt.axis("off")
        plt.show()


if __name__=="__main__":
    path="Desktop/Python/AI_ML/CNN/Train_images"
    img_size=64
    sample_images=2000
    test_ratio=0.2
    X_train,Y_train,X_test,Y_test,class_names=dl.load_and_split_dataset(path,img_size,sample_images,test_ratio)
    print(X_train.shape)
    print(Y_train.shape)
    config = {
    "activation_function": "relu",
    "output_activation_function": "softmax",
    "loss": "cross_entropy"
    }
    layers=[4096,1024,128,5]
    mlp_parameters=mlp.initialize_parameters(layers,config)
    cnn=CNN()
    mlp_parameters,cost_list=train(cnn,X_train,Y_train,mlp_parameters,config,0.001,12,64)
    filepath="model-saved"
    save_model(cnn,mlp_parameters,filepath)
    train_acc = compute_accuracy(cnn, mlp_parameters, config, X_test, Y_test)
    test_acc  = compute_accuracy(cnn, mlp_parameters, config, X_test, Y_test)
    show_prediction(cnn,mlp_parameters,config,X_test,Y_test,class_names,5)
    plot_confusion_matrix(cnn,mlp_parameters,config,X_train,Y_train,class_names)
    print(f"Train Accuracy: {train_acc:.2f}%")
    print(f"Test Accuracy:  {test_acc:.2f}%")