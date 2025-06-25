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
from sklearn.linear_model import RidgeCV
from SOG_Net.utilities import find_and_sort_neighbors_dimer
from SOG_Net.train import train_dimer
from SOG_Net.Neighbor import gen_coor_3d_species_three_body_dimer
from SOG_Net.DPSOG_dimer import DPSOG_dimer

print(tf.__version__)
import os 
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# 设置线程数
if tf.test.is_built_with_cuda():
    print("TensorFlow 已编译支持 GPU 加速")
else:
    print("TensorFlow 未编译支持 GPU 加速")
tf.keras.backend.set_floatx('float32')

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
Nsamples = 12                          # 100 number of samples 
descriptorNet = [20, 40, 80]        # [2, 4, 8, 16, 32] size of descriptor network       
fittingNet = [64, 64, 64]    # [32, 32, 32, 32, 32, 32] size of fitting network
epochsPerStair = 1000                      # 10 decay step of learning rate   
learningRate = 0.001                    # 0.001 initial learning rate
decayRate = 0.995                         # 0.95 decay rate of learning rate
Nepochs = [40000, 8000, 8000] #[20000, 400, 400]         # [200, 400, 800, 1600] epoch
batchSizeArray = [1,2,4]#[4, 8, 12]            # [8, 16, 32, 64] batchsize      
maxNumNeighs = 23                      # 120 最大近邻数 maximal number of neighbors
radious = 10                             # 8 近场截断半径 short-range interaction radious 
NpointsFourier = 31                      # 21 Fourier Mode 数量 the number of Fourier modes 
fftChannels = 1                          # 1 FFT频道 the number of FFT channels 
DataType = "Periodic"                    # "YukawaPeriodic" data type
L = 30                                   # 总的盒子边长
xLims = [0.0, L]                         # x方向的范围
Npoints = 23                          #总粒子数 1000 单个构型的
Test_type = "SLR2"                     #SR LR SLR SLR2

print("Test_type",Test_type)
model_load = "Sum-of-Gaussian_Neural_Network/model_and_loss/Dimer/best_model/CC_SR.h5"
#”/dssg/home/acct-matxzl/matxzl/Yajie/MDNN/Sum-of-Gaussian_Neural_Network/model_and_loss/20250527_170136_maxbest_model.h5"

# read data file
dataFile="Sum-of-Gaussian_Neural_Network/dataset/dimer0_CC_data.h5";
#print(dataFile)

nameScript=Test_type+"_Nsamples_" + str(Nsamples) +  "_NpointsFourier_" + str(NpointsFourier) + "_radious_" + str(radious) + "_decayRate_" + str(decayRate) + "_"

#Folder for saving loss, accuracy and model
saveFolder  = "Sum-of-Gaussian_Neural_Network/model_and_loss/Dimer/CC_"

# extracting the data
hf = h5py.File(dataFile, 'r')

pointsArray_Total = hf['points'][:,:Npoints]  #点的坐标数组
forcesArray_Total = hf['forces'][:,:Npoints]  #力的数组
energyArray_Total = hf['energy'][:]  #能量的数组
energyArray_Total = np.squeeze(energyArray_Total, axis=-1)
chargesArray_Total = hf['charges'][:,:Npoints]  #电荷的数组  
chargesArray_Total = np.squeeze(chargesArray_Total)
pointsArray_Total[pointsArray_Total < 0.0] += L
pointsArray_Total[pointsArray_Total >= L] -= L

#print(chargesArray_Total[0:13])
#print(pointsArray_Total.shape) # (2392, 25, 3)

index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]
pointsArray = pointsArray_Total[index]
forcesArray = forcesArray_Total[index]
energyArray = energyArray_Total[index]
chargesArray = chargesArray_Total[index]
chargesArray = np.squeeze(chargesArray)

pointsArray = tf.cast(pointsArray, tf.float32)
energyArray = tf.cast(energyArray, tf.float32)
forcesArray = tf.cast(forcesArray, tf.float32)
chargesArray = tf.cast(chargesArray, tf.float32)

#print(chargesArray.shape)
#print(chargesArray)
###################   预拟合   ##################

# print("energy:",energyArray.dtype)
X = np.zeros((Nsamples, 4), dtype=np.float32)
for i, val in enumerate([1, 2, 3, 4]):
    X[:, i] = np.sum(chargesArray == val, axis=1) 
Y = np.array(energyArray, dtype=np.float32) 
#print(X,chargesArray)
rcv = RidgeCV(alphas=np.geomspace(1e-8, 1e2, 10), fit_intercept=False)
rcv.fit(X, Y)
yp_base = rcv.predict(X)
y_diff = Y - yp_base
energyArray = y_diff.astype(np.float32)
# print("energy:",energyArray.dtype)


#print(y_diff.shape, energyArray.shape)
#print(energyArray, yp_base)
#################    预拟合结束   ################

Rinput = tf.Variable(pointsArray_Total, name="input", dtype = tf.float32) # 将数组信息输入tensorflow，可以通过反向传播和优化算法进行调整，pointsArray提供初始值的静态数据，“name”参数帮助在模型中标识和管理这个变量
Cinput = tf.Variable(chargesArray_Total, name="input", dtype = tf.float32)

# we only consider the first 100 
Rin = Rinput[:2,:,:] # 从Rinput中提取前100个构型，和相应的所有元素，赋值给Rin数组
Rinnumpy = Rin.numpy() # 将张量Rin的值转化为Rinnumpy数组，可以使用numpy的高效操作进行分析
Cin = Cinput[:2,:] # 从Rinput中提取前100个构型，和相应的所有元素，赋值给Rin数组
Cinnumpy = Cin.numpy() # 将张量Rin的值转化为Rinnumpy数组，可以使用numpy的高效操作进行分析

Idx = find_and_sort_neighbors_dimer(Rinnumpy, Cinnumpy, L, radious, maxNumNeighs)
#print(f"Hi, {Rin.shape}, {Idx.shape}")
# compute the neighbor list. shape:(Nsamples, Npoints and MaxNumneighs)
neighList = tf.Variable(Idx) #将生成的近邻表传递进去
genCoordinates, SRIJ = gen_coor_3d_species_three_body_dimer(Rin, neighList, L, 3*radious/4, radious)
 
# compute the generated coordinates
filter = tf.cast(tf.reduce_sum(tf.abs(genCoordinates), axis = -1)>0, tf.int32)
numNonZero =  tf.reduce_sum(filter, axis = [0,1,2]).numpy() # 非零元素个数 7941106
numTotal = genCoordinates.shape[0] * genCoordinates.shape[1] * genCoordinates.shape[2] # 总元素个数 12000000
#print(numNonZero, numTotal)

# 均值为0 方差为1
#print(genCoordinates.shape)
#print(tf.reduce_sum(genCoordinates, axis = 0, keepdims =True).numpy()[0])
av = tf.reduce_sum(genCoordinates, axis = [0,1,2], keepdims =True).numpy()[0]/numNonZero
std = np.sqrt((tf.reduce_sum(tf.square(genCoordinates - av), axis = [0,1,2], keepdims=True).numpy()[0] - av**2*(numTotal-numNonZero)) /numNonZero)
av = np.squeeze(av)
std = np.squeeze(std)
#print(av.shape,std) # 4个值

avTF = tf.constant(av, dtype=tf.float32)
stdTF = tf.constant(std, dtype=tf.float32)
## Define the model
model = DPSOG_dimer(Npoints, L, maxNumNeighs, descriptorNet, fittingNet, avTF, stdTF, NpointsFourier, fftChannels, xLims)

# for weight in model.trainable_weights:
#     print(f"Weight: {weight.name}, dtype: {weight.dtype}")
# for layer in model.layers:
#     new_weights = []
#     for weight in layer.get_weights():
#         # 只改变数据类型，保持形状不变
#         new_weight = tf.cast(weight, tf.float32)
#         new_weights.append(new_weight)
    
#     # 设置转换后的权重
#     layer.set_weights(new_weights)
# print("Change weights")

before_loss_time = time.time()
    
Rin2 = Rinput[:2,:,:] 
Cha2 = Cinput[:2,:]
Idx = find_and_sort_neighbors_dimer(Rin2.numpy(), Cha2.numpy(), L, radious, maxNumNeighs)

neigh_list2 = tf.Variable(Idx)

E, F = model(Rin2, Cha2, neigh_list2, radious, Test_type)
print(Rin2.dtype)

model.summary()

print("Training cycles in number of epochs")
print(Nepochs)
print("Training batch sizes for each cycle")
print(batchSizeArray)

errorlist_energy = []
errorlist_force = []
errorlist_energy_train = []
errorlist_force_train = []
max_errorlist = []
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
#dataFile="/dssg/home/acct-matxzl/matxzl/Yajie/MDNN/ELRC_3D/data/charge_charge_data.h5";

# extracting the data
hf = h5py.File(dataFile, 'r')

arr = list(range(13))
index_to_remove = index
index_test = [arr[i] for i in range(len(arr)) if i not in index_to_remove]
print(index_test)
pointsTest = hf['points'][index_test,:Npoints]  #点的坐标数组
forcesTest = hf['forces'][index_test,:Npoints]  #力的数组
energyTest = hf['energy'][index_test]  #能量的数组
energyTest = np.squeeze(energyTest, axis = -1)
chargesTest = hf['charges'][index_test,:Npoints]  #电荷的数组  

chargesTest=np.squeeze(chargesTest, axis=-1)
print(chargesTest)
pointsTest[pointsTest < 0] += L
pointsTest[pointsTest >= L] -= L

####################################   二次拟合能量   ###################################

X = np.zeros((len(index_test), 4), dtype=np.float32)
#print(X.shape)
for i, val in enumerate([1, 2, 3, 4]):
    X[:, i] = np.sum(chargesTest == val, axis=1) 
Y = np.array(energyTest, dtype=np.float32) 
print(X,chargesTest)
yp_base = rcv.predict(X)
y_diff = Y - yp_base
energyTest = y_diff.astype(np.float32)
print(energyTest)
print(yp_base)

######################################################################################

IdxTest = find_and_sort_neighbors_dimer(pointsTest, chargesTest, L, radious, maxNumNeighs)

neighListTest = tf.Variable(IdxTest)

#############################################  load model  ########################################
if Test_type == "SLR2":
  print("load Short model!")
  #model.load_weights('/dssg/home/acct-matxzl/matxzl/Yajie/MDNN/ELRC_3D/loss_accuracy_and_model_3d/CC/SR_model/SR_model.h5')
  model.load_weights(model_load)
  # 冻结指定的层
  #model.layerPyramid.trainable = False
  #model.layerPyramidDir.trainable = False
  #model.fittingNetwork.trainable = False
  #model.linfitNet.trainable = False
  #model.layerPyramid_Q.trainable = False
  #model.lrc_layer.trainable = False
  model.compile()
  for layer in model.layers:
    print(f"{layer.name} is trainable: {layer.trainable}")
  # layer = model.get_layer(name='nufft_layer_multi_channel3_dmixed_species__three_body_q')
  #weights, biases = layer.get_weights()
  #print(weights.shape,weights)
  #weights = np.linspace(0,1.54,12)
  #print(weights.shape,weights)
  #biases=0*biases
  #layer.set_weights([weights, biases])
for layer in model.layers:
    new_weights = []
    for weight in layer.get_weights():
        # 只改变数据类型，保持形状不变
        new_weight = tf.cast(weight, tf.float32)
        new_weights.append(new_weight)
    
    # 设置转换后的权重
    layer.set_weights(new_weights)
print("Change double weights for load model")
for weight in model.trainable_weights:
    print(f"Weight: {weight.name}, dtype: {weight.dtype}")

# r_input = pointsArray[:,:,:].numpy()
# Idx_r_input = find_and_sort_neighbors_dimer(r_input, chargesArray, L, radious, maxNumNeighs)

###################training loop ##################################
now = datetime.datetime.now()
time_name=now.strftime("%Y%m%d_%H%M%S")
min_test_F_err = np.Inf

for cycle, (epochs, batchSizeL) in enumerate(zip(Nepochs, batchSizeArray)):

  print('++++++++++++++++++++++++++++++', flush = True) 
  print('Start of cycle %d' % (cycle,))
  print('Total number of epochs in this cycle: %d'%(epochs,))
  print('Batch size in this cycle: %d'%(batchSizeL,))

  weightE = 1.0
  weightF = 1000.0

  # 重新组合为元组
  # print(pointsArray.dtype, energyArray.dtype, forcesArray.dtype, chargesArray.dtype)
  x_train = (pointsArray, energyArray, forcesArray, chargesArray)
  # print(x_train[1].dtype)
  train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
  train_dataset = train_dataset.shuffle(buffer_size=10000).batch(batchSizeL)
  
  # Iterate over epochs
  for epoch in range(epochs):
    #start = time.time()
    #print('============================', flush = True) 
    #print('Start of epoch %d' % (epoch,))
  
    loss_metric.reset_state()
    
    # Iterate over the batches of the dataset
    for step, x_batch_train in enumerate(train_dataset):
      # print(x_batch_train[1].dtype)
      start_epoch_time = time.time()

      Rinnumpy = x_batch_train[0].numpy()
      
      #print(f"That's ok")
      #print(Rinnumpy)
      
      #start = time.perf_counter()
      # print(Rinnumpy.shape, x_batch_train[3].numpy().shape)
      # print(Rinnumpy.dtype, x_batch_train[3].numpy().dtype)
      Idx = find_and_sort_neighbors_dimer(Rinnumpy, x_batch_train[3].numpy(), L, radious, maxNumNeighs)
      
      #end = time.perf_counter()
      #print(f"耗时: {end - start:.6f} 秒")

      #print(f"That's ok,1")
      neighList = tf.Variable(Idx)
      
      #before_loss_time = time.time()
      
      #if step % 10 ==0 :
       # print(f'List took {before_loss_time - start_epoch_time:.6f} seconds')
      #print(x_batch_train[3].dtype)

      #x_batch_train[0] 输入  x_batch_train[1] 能量输出  x_batch_train[2]力的输出
      #print("Train batch")
      #print(x_batch_train[0].dtype)

      #for weight in model.trainable_weights:
      #    print(f"Weight: {weight.name}, dtype: {weight.dtype}")
      
      #start = time.perf_counter()
      # print(x_batch_train[1].dtype)
      loss, gradients = train_dimer(model, optimizer, mse_loss_fn, x_batch_train[0], neighList, x_batch_train[3], radious, x_batch_train[1], x_batch_train[2], weightE, weightF, Test_type)
      #print(f"That's ok,2")
      loss_metric(loss)
      
      #end = time.perf_counter()
      #print(f"训练耗时: {end - start:.6f} 秒")

      #print(loss)
      # 打印每个变量及其对应的梯度
      #for var, grad in zip(model.trainable_variables, gradients):
      #  tf.print("Variable name:", var.name)
      #  tf.print("Gradient:", grad)
                           
      #after_loss_time = time.time()
      '''
      if step % 10 == 0:
        print(f'Train outloss took {after_loss_time - before_loss_time:.6f} seconds')

      if step % 10 == 0:
        print('step %s: total loss = %s' % (step, str(loss.numpy())))
  
      if step % 100 == 0:
        print('step %s: mean loss = %s' % (step, str(loss_metric.result().numpy())))
      #end_epoch_time = time.time()
      #print(f'Epoch took {end_epoch_time - start_epoch_time:.6f} seconds')
      '''

    #start = time.perf_counter()

    r_input = pointsArray[:,:,:]
    # print(r_input.shape, chargesArray[:,:].shape)
    # print(r_input.numpy().dtype, chargesArray[:,:].numpy().dtype)
    Idx = find_and_sort_neighbors_dimer(r_input.numpy(), chargesArray[:,:].numpy(), L, radious, maxNumNeighs)
    neighList = tf.Variable(Idx)
    pottrain, forcetrain = model(r_input, chargesArray[:,:], neighList, radious, Test_type)
    err_train = tf.sqrt(tf.reduce_mean(tf.square(forcetrain - forcesArray[:,:,:])))

    '''
    #print(f"That's ok,2")
    neighList_r_input = tf.Variable(Idx_r_input)
    pottrain, forcetrain = model(r_input, chargesArray, neighList_r_input, radious, Test_type)
    
    #print(forcetrain)
    #print(f"That's begin")
    err_train = tf.sqrt(tf.reduce_mean(tf.square(forcetrain - forcesArray[:,:,:].numpy())))
    #print(f"That's ok,1")
    #print(forcesArray[1,:,:])
    #AA = forcetrain - forcesArray[:,:,:]
    #print(forcetrain.shape, forcesArray.shape, AA.shape)
    '''
    #end = time.perf_counter()
    #print(f"测试训练耗时: {end - start:.6f} 秒")

    #errtrain = tf.sqrt(tf.reduce_sum(tf.square(forcetrain - forcesArray[:10,:,:])))/tf.sqrt(tf.reduce_sum(tf.square(forcetrain)))    
    
    err_ener_train = tf.sqrt(tf.reduce_mean(tf.square(pottrain - energyArray[:])))#/tf.sqrt(tf.reduce_sum(tf.square(pottrain)))
    #print(f"That's ok,2")
    #A = tf.expand_dims(pottrain, axis = -1) - energyArray[:] # (12,) (12, 1) (12, 1)
    #tf.print(pottrain.shape, energyArray.shape) # 
    #tf.print(pottrain - energyArray, summarize = -1)
    #tf.print(pottrain - energyArray[:,:], summarize = -1)
    #tf.print(pottrain, energyArray[:,:], "this is the outer", summarize=-1)

    #print(pottrain.shape(),potentialArray.shape())
    #print("Relative Error in the trained energy is " +str(err_ener_train.numpy()))
    
    #print("Relative denom" + str(tf.sqrt(tf.reduce_sum(tf.square(forcetrain))).numpy()))
    #print("Relative Error in the trained energy is " +str(err_ener_train.numpy()))
    #print("Relative Error in the trained forces is " +str(errtrain.numpy()))

    potPred, forcePred = model(pointsTest, chargesTest, neighListTest, radious, Test_type)

    err = tf.sqrt(tf.reduce_mean(tf.square(forcePred - forcesTest)))

    #err = tf.sqrt(tf.reduce_sum(tf.square(forcePred - forcesTest)))/tf.sqrt(tf.reduce_sum(tf.square(forcePred)))

    err_ener = tf.sqrt(tf.reduce_mean(tf.square(potPred - energyTest)))#/tf.sqrt(tf.reduce_sum(tf.square(potPred)))
    '''
    #tf.print(potPred - energyTest) 
    #potPred_Total, forcePred_Total = model(pointsArray_Total, chargesArray_Total, neighList, radious)
    diff1 = tf.abs(potPred - energyTest)
    
    #print(potPred, energyTest)
    
    diff2 = tf.abs(pottrain - energyArray[:12])

    #print(pottrain, energyArray[:12])

    # 将两组差值合并成一个张量
    all_diffs = tf.concat([diff1, diff2], axis=0)

    # 取绝对值中的最大值
    max_diff = tf.reduce_max(all_diffs).numpy()

    if err_ener_train+err_ener < min_test_F_err:
      min_test_F_err = err_ener_train+err_ener
    if err_ener_train<1e-2 and err_ener<1e-2:
      model.save_weights(saveFolder+time_name0+'_tol11.h5', save_format='h5')
      if err_ener_train<2e-2 and err_ener<2e-2:
        model.save_weights(saveFolder+time_name0+'_tol22.h5', save_format='h5')
      if err_ener_train<5e-3 and err_ener<5e-3:
        model.save_weights(saveFolder+time_name0+'_tol33.h5', save_format='h5')
      if err_ener_train<1e-4 and err_ener<1e-4:
        model.save_weights(saveFolder+time_name0+'_tol44.h5', save_format='h5')

    if max_diff < min_test_F_err:
      min_test_F_err = max_diff
      print(saveFolder+time_name0+'_maxbest_model.h5')
      model.save_weights(saveFolder+time_name0+'_maxbest_model.h5', save_format='h5')
      print("Max Error in the energy is " +str(min_test_F_err))
      #print("Relative Error in the trained forces is " +str(err_train.numpy()))
      #print("Relative Error in the trained energy is " +str(err_ener_train.numpy()))
      #print("Relative Error in the test forces is " +str(err.numpy()))
      #print("Relative Error in the test energy is " +str(err_ener.numpy()))

    #end = time.time()
    #print('time elapsed %.4f'%(end - start))
    
    # save the error
    max_errorlist.append(max_diff) #此处
    with open(saveFolder+time_name0+'_error_max_'+nameScript+'.csv','w') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(errorlist_force)
    errorlist_force.append(err.numpy()) #此处
    with open(saveFolder+time_name0+'_error_force_'+nameScript+'.csv','w') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(errorlist_force)
    errorlist_energy.append(err_ener.numpy()) #此处
    with open(saveFolder+time_name0+'_error_energy_'+nameScript+'.csv','w') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(errorlist_energy) 
    errorlist_force_train.append(err_train.numpy())
    with open(saveFolder+time_name0+'_error_force_train_'+nameScript+'.csv','w') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(errorlist_force_train)
    errorlist_energy_train.append(err_ener_train.numpy()) #此处
    with open(saveFolder+time_name0+'_error_energy_train_'+nameScript+'.csv','w') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(errorlist_energy_train) 
    # mean loss saved in the metric
    meanLossStr = str(loss_metric.result().numpy())
    # decay learning rate 
    
    #lrStr = str(optimizer._decayed_lr('float32').numpy())
    lr_value = optimizer.learning_rate(optimizer.iterations).numpy()#(optimizer.learning_rate).numpy()
    lrStr = str(lr_value)
    
    if epoch%100 ==0:
      print('epoch %s: mean loss = %s  learning rate = %s'%(epoch,meanLossStr,lrStr))
    
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
  '''

    #if err_ener_train+err_ener < min_test_F_err:
     # min_test_F_err = err_ener_train+err_ener
    if err_ener_train<2e-2 and err_ener<2e-2:
      model.save_weights(saveFolder+time_name0+'_tol2.h5', save_format='h5')
      if err_ener_train<1e-2 and err_ener<2e-2:
        model.save_weights(saveFolder+time_name0+'_tol12.h5', save_format='h5')
      if err_ener_train<2e-2 and err_ener<1e-2:
        model.save_weights(saveFolder+time_name0+'_tol21.h5', save_format='h5')
      print("Relative Error in the trained forces is " +str(err_train.numpy()))
      print("Relative Error in the trained energy is " +str(err_ener_train.numpy()))
      print("Relative Error in the test forces is " +str(err.numpy()))
      print("Relative Error in the test energy is " +str(err_ener.numpy()))

    #end = time.time()
    #print('time elapsed %.4f'%(end - start))
    
    # save the error
    errorlist_force.append(err.numpy()) #此处
    with open(saveFolder+time_name0+'_error_force_'+nameScript+'.csv','w') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(errorlist_force)
    errorlist_energy.append(err_ener.numpy()) #此处
    with open(saveFolder+time_name0+'_error_energy_'+nameScript+'.csv','w') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(errorlist_energy) 
    errorlist_force_train.append(err_train.numpy())
    with open(saveFolder+time_name0+'_error_force_train_'+nameScript+'.csv','w') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(errorlist_force_train)
    errorlist_energy_train.append(err_ener_train.numpy()) #此处
    with open(saveFolder+time_name0+'_error_energy_train_'+nameScript+'.csv','w') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(errorlist_energy_train) 
    # mean loss saved in the metric
    meanLossStr = str(loss_metric.result().numpy())
    # decay learning rate 
    
    #lrStr = str(optimizer._decayed_lr('float32').numpy())
    lr_value = optimizer.learning_rate(optimizer.iterations).numpy()#(optimizer.learning_rate).numpy()
    lrStr = str(lr_value)
    
    if epoch%100 ==0:
      print('epoch %s: mean loss = %s  learning rate = %s'%(epoch,meanLossStr,lrStr))
    
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



