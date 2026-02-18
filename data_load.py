import os
import cv2
import numpy as np

def load_and_split_dataset(root_path,img_size=64,sample_images=4000,test_ratio=0.2):
    X=[]
    Y=[]

    class_name=sorted(os.listdir(root_path))
    class_to_idx={cls: i for i,cls in enumerate(class_name)}

    count=0

    for cls in class_name:
        class_folder=os.path.join(root_path,cls)

        for img_name in os.listdir(class_folder):
            img_path=os.path.join(class_folder,img_name)

            img=cv2.imread(img_path)
            img=cv2.resize(img,(img_size,img_size))
            img=img.astype(np.float32)/255.0

            X.append(img)
            Y.append(class_to_idx[cls])

            count+=1
            if sample_images and count>=sample_images:
                break

        if sample_images and count>=sample_images:
            break

    X=np.array(X)
    X=X.transpose(0,3,1,2)

    num_classes=len(class_name)
    Y_onehot=np.zeros((num_classes,len(Y)))
    Y_onehot[Y,np.arange(len(Y))]=1

    N=X.shape[0]

    indices=np.random.permutation(N)
    test_size=int(N*test_ratio)

    test_idx=indices[:test_size]
    train_idx=indices[test_size:]

    X_train=X[train_idx]
    Y_train=Y_onehot[:,train_idx]

    X_test=X[test_idx]
    Y_test=Y_onehot[:,test_idx]

    mean=np.mean(X_train,axis=(0,2,3),keepdims=True)
    std=np.std(X_train,axis=(0,2,3),keepdims=True)

    X_train=(X_train-mean)/(std+1e-8)
    X_test=(X_test-mean)/(std+1e-8)

    return X_train,Y_train,X_test,Y_test,class_name