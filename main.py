import trainer 
import numpy as np
import os

AK_set = [(3,0),(5,0),(10,0),(15,0),(20,0),(25,0),(30,0),(35,0),(40,0),(45,0),(50,0),
            (10,5),(10,10),(10,15),(10,20),(10,25),(10,30),(10,35),(10,40),(10,45),(10,50)]

print('Testing with OvA mode')
manager = trainer.manger(mode='ova')
manager.max_runs_ = 10
manager.max_epochs_ = 30
manager.batch_size_ = 16
for A, K in AK_set:
    print('\tA:%d, K:%d'%(A,K))
    manager.run_given_AK(A_size=A,K_size=K)

# print('Testing with Autoencoder mode')
# manager = trainer.manger(mode='autoencoder')
# manager.max_runs_ = 10
# manager.max_epochs_ = 50
# manager.batch_size_ = 16
# for A, K in AK_set:
#     print('\tA:%d, K:%d'%(A,K))
#     manager.run_given_AK(A_size=A,K_size=K)