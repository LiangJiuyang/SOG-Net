import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import os.path
from os import path
import h5py
import sys
import json
import time
import csv
import datetime  #时间戳



from SOG_Net.utilities import gen_coor_3d
from SOG_Net.train import train_pointcharge
from SOG_Net.DPSOG_pointcharge import DPSOG_pointcharge

import os 
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

if tf.test.is_built_with_cuda():
    print("TensorFlow 已编译支持 GPU 加速")
else:
    print("TensorFlow 未编译支持 GPU 加速")

# 检查当前 TensorFlow 是否在 GPU 上执行
if tf.test.gpu_device_name():
    print("当前 TensorFlow 正在使用 GPU:", tf.test.gpu_device_name())
else:
    print("当前 TensorFlow 没有在 GPU 上执行")
gpus = tf.config.experimental.list_physical_devices('GPU')

###  获取时间戳  ###
now = datetime.datetime.now()
time_name0=now.strftime("%Y%m%d_%H%M%S")

# Input Data
# here we assume the data is generated within some cells. The number of cells in
# each dimension is "Ncells". "Np" shows the number of particles in per cell. 
# For simiplicity, we assume they are generated randomly uniformly.   
Nsamples = 1000                        # 100 number of samples 
descriptorNet = [2, 4, 8, 16, 32]        # [2, 4, 8, 16, 32] size of descriptor network       
fittingNet = [32, 32, 32, 32, 32, 32]    # [32, 32, 32, 32, 32, 32] size of fitting network
epochsPerStair = 100                      # 10 decay step of learning rate   
learningRate = 0.001                     # 0.001 initial learning rate
decayRate = 0.99                         # 0.95 decay rate of learning rate
Nepochs = [200, 400, 800, 1600]          # [200, 400, 800, 1600] epoch
batchSizeArray = [8, 16, 32, 64]            # [8,16,32,64] batchsize      
maxNumNeighs = 20                       # 120 最大近邻数 maximal number of neighbors
radious = 1.5                             # 8 近场截断半径 short-range interaction radious 
NpointsFourier = 17                      # 21 Fourier Mode 数量 the number of Fourier modes 
fftChannels = 1                          # 1 FFT频道 the number of FFT channels 
DataType = "Periodic"                    # "YukawaPeriodic" data type
L = 10                                   # 总的盒子边长
xLims = [0.0, L]                         # x方向的范围
Npoints = 200                           #总粒子数 1000 单个构型的

# read data file
dataFile="Sum-of-Gaussian_Neural_Network/dataset/pointcharge_data_train.h5";
print(dataFile)

nameScript="_Nsamples_" + str(Nsamples) +  "_NpointsFourier_" + str(NpointsFourier) + "_radious_" + str(radious) + "_decayRate_" + str(decayRate) 

#Folder for saving loss, accuracy and model
saveFolder  = "Sum-of-Gaussian_Neural_Network/model_and_loss/pointcharge/"

# extracting the data
hf = h5py.File(dataFile, 'r')

pointsArray = hf['points'][:]  #点的坐标数组
forcesArray = hf['forces'][:]  #力的数组
energyArray = hf['energy'][:]  #能量的数组
chargesArray = hf['charges'][:]  #电荷的数组  

pointsArray[pointsArray < 0.0] += L
pointsArray[pointsArray >= L] -= L

pointsArray = np.transpose(pointsArray, axes=(2, 1, 0))
forcesArray = np.transpose(forcesArray, axes=(2, 1, 0))
energyArray = np.transpose(energyArray, axes=(1, 0))
chargesArray = np.transpose(chargesArray, axes=(1, 0))

print(forcesArray.shape)
print(energyArray.shape)
print(chargesArray.shape)

Rinput = tf.Variable(pointsArray, name="input", dtype = tf.float32) # 将数组信息输入tensorflow，可以通过反向传播和优化算法进行调整，pointsArray提供初始值的静态数据，“name”参数帮助在模型中标识和管理这个变量
Cinput = tf.Variable(chargesArray, name="input", dtype = tf.float32)

# we only consider the first 100 
Rin = Rinput[:100,:,:] # 从Rinput中提取前100个构型，和相应的所有元素，赋值给Rin数组
Rinnumpy = Rin.numpy() # 将张量Rin的值转化为Rinnumpy数组，可以使用numpy的高效操作进行分析

#This is the new implementation
n_samples, n_points, dimension = Rinnumpy.shape
Idx = np.zeros((n_samples, n_points, maxNumNeighs), dtype=np.int32)-1

for i in range(n_samples):
    r_sample = Rinnumpy[i]
    tree = cKDTree(r_sample, boxsize=[L,L,L])
    r_list = tree.query_ball_point(r_sample,radious)
    r_list=[[elem for elem in row if elem!=i] for i,row in enumerate(r_list)]    
    for j,row in enumerate(r_list):
      Idx[i,j,:len(row)]=row

#Idx = comput_inter_list(Rinnumpy, L,  radious, maxNumNeighs) # 计算近邻表

# compute the neighbor list. shape:(Nsamples, Npoints and MaxNumneighs)
neighList = tf.Variable(Idx) #将生成的近邻表传递进去

genCoordinates = gen_coor_3d(Rin, neighList, L)
# compute the generated coordinates
filter = tf.cast(tf.reduce_sum(tf.abs(genCoordinates), axis = -1)>0, tf.int32)
numNonZero =  tf.reduce_sum(filter, axis = 0).numpy() # 非零元素个数 7941106
numTotal = genCoordinates.shape[0] # 零元素个数 12000000

# 均值为0 方差为1
av = tf.reduce_sum(genCoordinates, axis = 0, keepdims =True).numpy()[0]/numNonZero
std = np.sqrt((tf.reduce_sum(tf.square(genCoordinates - av), axis = 0, keepdims=True).numpy()[0] - av**2*(numTotal-numNonZero)) /numNonZero)

avTF = tf.constant(av, dtype=tf.float32)
stdTF = tf.constant(std, dtype=tf.float32)
## Define the model
model = DPSOG_pointcharge(Npoints, L, maxNumNeighs, descriptorNet, fittingNet, avTF, stdTF, NpointsFourier, fftChannels, xLims)

# quick run of the model to check that it is correct.
#This is the new implementation
Rin2 = Rinput[:2,:,:] 
Cha2 = Cinput[:2,:]
n_samples, n_points, dimension = Rin2.shape
Idx = np.zeros((n_samples, n_points, maxNumNeighs), dtype=np.int32)-1
for i in range(n_samples):
    r_sample = Rinnumpy[i]
    tree = cKDTree(r_sample, boxsize=[L,L,L])
    r_list = tree.query_ball_point(r_sample,radious)
    r_list=[[elem for elem in row if elem!=i] for i,row in enumerate(r_list)]    
    for j,row in enumerate(r_list):
      Idx[i,j,:len(row)]=row

neigh_list2 = tf.Variable(Idx)

E, F = model(Rin2, Cha2, neigh_list2)

model.summary()

print("Training cycles in number of epochs")
print(Nepochs)
print("Training batch sizes for each cycle")
print(batchSizeArray)

errorlist = []
losslist = []

### optimization parameters ##
mse_loss_fn = tf.keras.losses.MeanSquaredError()

initialLearningRate = learningRate
lrSchedule = tf.keras.optimizers.schedules.ExponentialDecay(
             initialLearningRate,
             decay_steps=(Nsamples//batchSizeArray[0])*epochsPerStair,
             decay_rate=decayRate,
             staircase=True)

optimizer = tf.keras.optimizers.Adam(learning_rate=lrSchedule)

loss_metric = tf.keras.metrics.Mean()

# read data file of test set
dataFile="Sum-of-Gaussian_Neural_Network/dataset/pointcharge_data_test.h5";
print(dataFile)

# extracting the data
hf = h5py.File(dataFile, 'r')

pointsTest = hf['points'][:]  #点的坐标数组
forcesTest = hf['forces'][:]  #力的数组
energyTest = hf['energy'][:]  #能量的数组
chargesTest = hf['charges'][:]  #电荷的数组  

pointsTest[pointsTest < 0] += L
pointsTest[pointsTest >= L] -= L

pointsTest = np.transpose(pointsTest, axes=(2, 1, 0))
forcesTest = np.transpose(forcesTest, axes=(2, 1, 0))
energyTest = np.transpose(energyTest, axes=(1, 0))
chargesTest = np.transpose(chargesTest, axes=(1, 0))

#print(pointsTest.shape)
#print(energyTest.shape)
#print(chargesTest.shape)

#This is the new implementation
n_samples, n_points, dimension = pointsTest.shape
IdxTest = np.zeros((n_samples, n_points, maxNumNeighs), dtype=np.int32)-1
for i in range(n_samples):
    r_sample = pointsTest[i]
    tree = cKDTree(r_sample, boxsize=[L,L,L])
    r_list = tree.query_ball_point(r_sample,radious)
    r_list=[[elem for elem in row if elem!=i] for i,row in enumerate(r_list)]    
    for j,row in enumerate(r_list):
      IdxTest[i,j,:len(row)]=row

#print(maxNumNeighs)

neighListTest = tf.Variable(IdxTest)

#model.load_weights('./loss_accuracy_and_model/20240613_122208my_model_3.h5')
#rin_test = tf.Variable(points_test, dtype=tf.float32)
#forces_test = tf.Variable(forces_test, dtype=tf.float32)
###################training loop ##################################
now = datetime.datetime.now()
time_name=now.strftime("%Y%m%d_%H%M%S")

for cycle, (epochs, batchSizeL) in enumerate(zip(Nepochs, batchSizeArray)):

  print('++++++++++++++++++++++++++++++', flush = True) 
  print('Start of cycle %d' % (cycle,))
  print('Total number of epochs in this cycle: %d'%(epochs,))
  print('Batch size in this cycle: %d'%(batchSizeL,))

  weightE = 0.0
  weightF = 1.0

  x_train = (pointsArray, energyArray, forcesArray, chargesArray)

  train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
  train_dataset = train_dataset.shuffle(buffer_size=10000).batch(batchSizeL)

  # Iterate over epochs
  for epoch in range(epochs):
    start = time.time()
    print('============================', flush = True) 
    print('Start of epoch %d' % (epoch,))
  
    loss_metric.reset_state()
    
    # Iterate over the batches of the dataset
    for step, x_batch_train in enumerate(train_dataset):
      start_epoch_time = time.time()
      
      #print(x_batch_train[0].shape)

      Rinnumpy = x_batch_train[0].numpy()

      #This is the old implementation
      #Idx = comput_inter_list(Rinnumpy, L, radious, maxNumNeighs)
      
      #This is the new implementation
      n_samples, n_points, dimension = Rinnumpy.shape
      Idx = np.zeros((n_samples, n_points, maxNumNeighs), dtype=np.int32)-1
      for i in range(n_samples):
        r_sample=Rinnumpy[i]
        #print(tf.reduce_max(r_sample))
        #print(tf.reduce_min(r_sample))
        tree=cKDTree(r_sample,boxsize=[L,L,L])
        r_list = tree.query_ball_point(r_sample,radious)
        r_list=[[elem for elem in row if elem!=i] for i,row in enumerate(r_list)] 
        for j,row in enumerate(r_list):
          Idx[i,j,:len(row)]=row

      neighList = tf.Variable(Idx)
      
      before_loss_time = time.time()
      
      # if step % 10 ==0 :
      #   print(f'List took {before_loss_time - start_epoch_time:.6f} seconds')

      #x_batch_train[0] 输入  x_batch_train[1] 能量输出  x_batch_train[2]力的输出
      loss,_ = train_pointcharge(model, optimizer, mse_loss_fn, x_batch_train[0], neighList, x_batch_train[3], x_batch_train[1], x_batch_train[2], weightE, weightF)
                           
      after_loss_time = time.time()
      
      # if step % 10 == 0:
      #   print(f'Train outloss took {after_loss_time - before_loss_time:.6f} seconds')

      loss_metric(loss)
      if step % 10 == 0:
        print('step %s: total loss = %s' % (step, str(loss.numpy())))
  
      if step % 100 == 0:
        print('step %s: mean loss = %s' % (step, str(loss_metric.result().numpy())))
      #end_epoch_time = time.time()
      #print(f'Epoch took {end_epoch_time - start_epoch_time:.6f} seconds')
    
    r_input = pointsArray[:10,:,:]
    n_samples, n_points, dimension = r_input.shape
    Idx = np.zeros((n_samples, n_points, maxNumNeighs), dtype=np.int32)-1
    for i in range(n_samples):
      r_sample = r_input[i]
      tree = cKDTree(r_sample, boxsize=[L,L,L])
      r_list = tree.query_ball_point(r_sample,radious)
      r_list=[[elem for elem in row if elem!=i] for i,row in enumerate(r_list)]  
      for j,row in enumerate(r_list):
        Idx[i,j,:len(row)]=row
    neighList = tf.Variable(Idx)
    #print(neighList.shape)
    
    pottrain, forcetrain = model(r_input, chargesArray[:10,:], neighList)

    errtrain = tf.sqrt(tf.reduce_sum(tf.square(forcetrain - forcesArray[:10,:,:])))\
               /tf.sqrt(tf.reduce_sum(tf.square(forcetrain)))

    err_ener_train=tf.sqrt(tf.reduce_sum(tf.square(pottrain-energyArray[:10,:])))/tf.sqrt(tf.reduce_sum(tf.square(pottrain)))

    #print(pottrain.shape(),potentialArray.shape())
    #print("Relative Error in the trained energy is " +str(err_ener_train.numpy()))
    print("Relative Error in the trained forces is " +str(errtrain.numpy()))

    potPred, forcePred = model(pointsTest, chargesTest, neighListTest)
    print(forcePred.shape,forcesTest.shape)
    err = tf.sqrt(tf.reduce_sum(tf.square(forcePred - forcesTest)))/tf.sqrt(tf.reduce_sum(tf.square(forcePred)))
    print(potPred.shape,energyTest.shape)
    err_ener=tf.sqrt(tf.reduce_sum(tf.square(potPred - energyTest)))/tf.sqrt(tf.reduce_sum(tf.square(potPred)))

    #print("Relative Error in the energy is " +str(err_ener.numpy()))
    print("Relative Error in the forces is " +str(err.numpy()))

    end = time.time()
    print('time elapsed %.4f'%(end - start))
    
    # save the error
    errorlist.append(err.numpy())
    with open(saveFolder+time_name0+'_error_'+nameScript+'.csv','w') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(errorlist)
        
    # mean loss saved in the metric
    meanLossStr = str(loss_metric.result().numpy())
    # decay learning rate 
    
    #lrStr = str(optimizer._decayed_lr('float32').numpy())
    lr_value = optimizer.learning_rate(optimizer.iterations).numpy()#(optimizer.learning_rate).numpy()
    lrStr = str(lr_value)
    
    print('epoch %s: mean loss = %s  learning rate = %s'%(epoch,
                                                          meanLossStr,
                                                          lrStr))
    
    # save the loss
    losslist.append(loss_metric.result().numpy())
    with open(saveFolder+time_name0+'_loss_'+nameScript+'.csv','w') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(losslist)
    
  # 获取目录路径
  #directory = os.path.dirname(checkFile)

  # 如果目录不存在，则创建目录
  #if not os.path.exists(directory):
  #  os.makedirs(directory)
  print("saving the weights")
  ###  获取时间戳  ###
  str_num = str(cycle)

  model.save_weights(saveFolder+time_name+'my_model_'+str_num+'.h5', save_format='h5')



