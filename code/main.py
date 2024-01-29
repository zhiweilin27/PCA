import torch
from helper import load_data
from solution import PCA, AE, frobeniu_norm_error
import numpy as np
import os


def test_pca(A, p):
    pca = PCA(A, p)
    Ap, G = pca.get_reduced()
    A_re = pca.reconstruction(Ap)
    error = frobeniu_norm_error(A, A_re)
    print('PCA-Reconstruction error for {k} components is'.format(k=p), error)
    return G

def test_ae(A, p):
    model = AE(d_hidden_rep=p)
    model.train(A, A, 64, 1000)
    A_re = model.reconstruction(A)
    final_w = model.get_params()
    error = frobeniu_norm_error(A, A_re)
    print('AE-Reconstruction error for {k}-dimensional hidden representation is'.format(k=p), error)
    return final_w

if __name__ == '__main__':
    dataloc = "/Users/zhiweilin/Desktop/AMS 691.03_HW3/PCA/data/USPS.mat"
    A = load_data(dataloc)
    A = A.T
    ## Normalize A
    A = A/A.max()

    ### YOUR CODE HERE
    # Note: You are free to modify your code here for debugging and justifying your ideas for 5(f)
    ps = [32,64,128]
    for p in ps:
        #G = test_pca(A, p)
        final_w = test_ae(A, p)
#         err =frobeniu_norm_error(G,final_w)
#         print('for p=d={k}, relation between the projection matrix G in PCA and the optimized weight matrix W in AE is'.format(k=p) ,err)
#         err2 = frobeniu_norm_error(G.T@G,final_w.T@final_w)
#         print('for p=d={k}, relation between the projection matrix G^TG in PCA and the optimized weight matrix W^TW in AE is'.format(k=p) ,err2)
    ### END YOUR CODE 
