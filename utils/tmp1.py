import numpy as np 

def one_hot_encode(label, label_num):
    # one-hot encoded label
    label_indices=np.zeros(label_num)
    label_indices[label]=1
    return label_indices

if __name__=="__main__":
    label_num=4
    label=2
    aa=one_hot_encode(label,label_num)
    b= np.full((3),False)