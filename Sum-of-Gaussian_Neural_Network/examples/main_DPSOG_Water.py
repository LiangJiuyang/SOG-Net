import tensorflow as tf
import numpy as np
import os.path
from os import path
import h5py
import time
import csv
import datetime  #时间戳
from SOG_Net.utilities import find_and_sort_neighbors_water
from SOG_Net.train import train_Water

from SOG_Net.DPSOG_Water import DPSOG_Water
#from tensorflow.keras import mixed_precision

import os 
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
# 设置线程数
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

#mixed_precision.set_global_policy('float32')

# 检查当前的全局精度策略
#print("当前全局精度策略:", mixed_precision.global_policy().name)

###  获取时间戳  ###
now = datetime.datetime.now()
time_name0=now.strftime("%Y%m%d_%H%M%S")

################# Input Data ################
# here we assume the data is generated within some cells. The number of cells in
# each dimension is "Ncells". "Np" shows the number of particles in per cell. 
# For simiplicity, we assume they are generated randomly uniformly.   
Nsamples = 1000                          # 100 number of samples 
descriptorNet = [25, 50, 100]        # [2, 4, 8, 16, 32] size of descriptor network       
fittingNet = [120, 120, 120]    # [32, 32, 32, 32, 32, 32] size of fitting network
epochsPerStair = 1000                      # 10 decay step of learning rate   
learningRate = 0.001                    # 0.001 initial learning rate
decayRate = 0.99                         # 0.95 decay rate of learning rate
Nepochs = [40000, 40000]         # [200, 400, 800, 1600] epoch
batchSizeArray = [50, 100]            # [8,16,32,64] batchsize      
maxNumNeighs_O_A = 25            # A部分近邻的最大氧原子数
maxNumNeighs_H_A = 50            # A部分近邻的最大氢原子数
maxNumNeighs_O_R = 60            # R部分近邻的最大氧原子数
maxNumNeighs_H_R = 120           # R部分近邻的最大氢原子数
radious_A = 5                             # 8 近场截断半径 short-range interaction radious
radious_R = 7 
NpointsFourier = 5                       # 21 Fourier Mode 数量 the number of Fourier modes 
fftChannels = 1                          # 1 FFT频道 the number of FFT channels 
DataType = "Periodic"                     # "YukawaPeriodic" data type
L = 14.4088                          # 盒子边长
Npoints = 300                          # 总粒子数 1000 单个构型的
select_neuron = 16                   # 选择的不变特征数量
Test_type = "SLR2"                     # SR LR SLR SLR2
Energy_rate = 0.0
Force_rate = 1.0
Load_short = 0                      # load/unload the existing model
inner_factor_A = 1.0/8.0              # 开始软化的位置
inner_factor_R = 1.0/14.0             # 开始软化的位置

model_load = "Sum-of-Gaussian_Neural_Network/dataset/pointcharge_data_test.h5"
mirrored_strategy = tf.distribute.MirroredStrategy()

# read data file
dataFile = "../dataset/water_1900_data.h5"
hf = h5py.File(dataFile, 'r')
pointsArray_Total = hf['points'][:]  #点的坐标数组
forcesArray_Total = hf['forces'][:]  #力的数组
energyArray_Total = hf['energy'][:]  #能量的数组
energyArray_Total = np.squeeze(energyArray_Total, axis=-1)
chargesArray_Total = hf['charges'][:]  #电荷的数组
chargesArray_Total = np.squeeze(chargesArray_Total)
nameScript="_Nsamples_" + str(Nsamples) +  "_NpointsFourier_" + str(NpointsFourier) + "_radious_A_" + str(radious_A) + "_radious_R_" + str(radious_R) + "_decayRate_" + str(decayRate) + "_water_" + Test_type + "_energyRate_" + str(Energy_rate) + "_forceRate_" + str(Force_rate) + "_inner_factor_A_" + str(inner_factor_A) + "_inner_factor_R_" + str(inner_factor_R)
#Folder for saving loss, accuracy and model
#saveFolder  = "/dssg/home/acct-matxzl/matxzl/Yajie/MDNN/ELRC_3D/loss_accuracy_and_model_3d/WaterCC/WaterCC_deepmd/"
saveFolder  = "model_and_loss/water/"

print(pointsArray_Total.shape)

# extracting the data: 90% of dataset
pointsArray = pointsArray_Total[:(Nsamples//10*9),:Npoints]
forcesArray = forcesArray_Total[:(Nsamples//10*9),:Npoints]
energyArray = energyArray_Total[:(Nsamples//10*9)]
chargesArray = chargesArray_Total[:(Nsamples//10*9),:Npoints]

#print(chargesArray[1,:]) # 1: O  2: H
#print(f"The size of training set is {(Nsamples//10*9)}; The size of testing set is {Nsamples-(Nsamples//10*9)}")

# 将数组信息输入tensorflow，可以通过反向传播和优化算法进行调整，pointsArray提供初始值的静态数据，“name”参数帮助在模型中标识和管理这个变量
Rinput = tf.Variable(pointsArray, name="input", dtype = tf.float32)
Cinput = tf.Variable(chargesArray, name="input", dtype = tf.float32) 

# read data file of test set: 10% of dataset
pointsTest = pointsArray_Total[(Nsamples//10*9):Nsamples,:Npoints]  #点的坐标数组
forcesTest = forcesArray_Total[(Nsamples//10*9):Nsamples,:Npoints]  #力的数组
energyTest = energyArray_Total[(Nsamples//10*9):Nsamples]  #能量的数组
energyTest = np.squeeze(energyTest)
chargesTest = chargesArray_Total[(Nsamples//10*9):Nsamples,:Npoints]  #电荷的数组  
chargesTest = np.squeeze(chargesTest)

#print(chargesTest.shape, chargesArray.shape, energyArray.shape, energyTest.shape)

## Define the model
model = DPSOG_Water(Npoints, maxNumNeighs_O_A, maxNumNeighs_H_A, maxNumNeighs_O_R, maxNumNeighs_H_R, L, descriptorNet, fittingNet, NpointsFourier, fftChannels)

## Test Model ###############################################################################################################################
Rin2 = pointsArray[:10] 
Cha2 = chargesArray[:10]
#before_loss_time = time.time()
Idx_O_A, Idx_H_A, Idx_O_R, Idx_H_R = find_and_sort_neighbors_water(Rin2, Cha2, L, radious_A, maxNumNeighs_O_A, maxNumNeighs_H_A, radious_R, maxNumNeighs_O_R, maxNumNeighs_H_R)
# print(Idx_O_A.shape, Idx_H_A.shape, Idx_O_R.shape, Idx_H_R.shape)
# (2, 300, 16) (2, 300, 32) (2, 300, 60) (2, 300, 120)
# 进一步区分O和H的target
mask_Cha2_O = tf.equal(Cha2, 1)  # 创建形状为 (2, 300) 的布尔掩码
mask_Cha2_H = tf.equal(Cha2, 2)  # 创建形状为 (2, 300) 的布尔掩码
for i in ["O", "H"]: 
        for j in ["O", "H"]:
            for k in ["A", "R"]:
                exec(f"Idx_{i}_{j}_{k} = []")
for m in range(Rin2.shape[0]):  # 遍历批次
    for i in ["O", "H"]:
        exec(f"batch_mask_{i} = mask_Cha2_{i}[m]") # 当前批次掩码 
        for j in ["O", "H"]:
            for k in ["A", "R"]:
                exec(f"selected_{i}_{j}_{k} = tf.boolean_mask(Idx_{j}_{k}[m], batch_mask_{i})")
                exec(f"Idx_{i}_{j}_{k}.append(selected_{i}_{j}_{k})")
for i in ["O", "H"]:
    for j in ["O", "H"]:
        for k in ["A", "R"]:
            exec(f"Idx_{i}_{j}_{k} =  tf.stack(Idx_{i}_{j}_{k}, axis=0)")
            exec(f"print(Idx_{i}_{j}_{k}.shape)")
# (10, 100, 25) (10, 100, 80) (10, 100, 50) (10, 100, 160) (10, 200, 25) (10, 200, 80) (10, 200, 50) (10, 200, 160)

LArray2 = tf.Variable(L, name="input", dtype = tf.float32)
E, F = model(Rin2, Cha2, tf.cast(LArray2, dtype = tf.float32), select_neuron, inner_factor_A, radious_A, Idx_O_O_A, Idx_H_O_A, Idx_O_H_A, Idx_H_H_A, inner_factor_R, radious_R, Idx_O_O_R, Idx_H_O_R, Idx_O_H_R, Idx_H_H_R, Test_type)
#print(F[0,1:5,:])
model.summary()

# All of these variables are tensorflow tensors
# 创建布尔掩码   

#end = time.time()
#print('time 22 elapsed %.4f'%(end - before_loss_time))
before_loss_time = time.time()
for i in range(100):
    E, F = model(Rin2, Cha2, tf.cast(LArray2, dtype = tf.float32), select_neuron, inner_factor_A, radious_A, Idx_O_O_A, Idx_H_O_A, Idx_O_H_A, Idx_H_H_A, inner_factor_R, radious_R, Idx_O_O_R, Idx_H_O_R, Idx_O_H_R, Idx_H_H_R, Test_type)
#print(F.shape)
end = time.time()
print('time 22 elapsed %.4f'%(end - before_loss_time))

## Calculate the neighborlist 用Numpy比Tensorflow变量要快很多 ###############################################################################
#before_loss_time = time.time()
Idx_O_A, Idx_H_A, Idx_O_R, Idx_H_R = find_and_sort_neighbors_water(pointsArray, chargesArray, L, radious_A, maxNumNeighs_O_A, maxNumNeighs_H_A, radious_R, maxNumNeighs_O_R, maxNumNeighs_H_R)
print(Idx_O_A.shape, Idx_H_A.shape, Idx_O_R.shape, Idx_H_R.shape)
#(270, 300, 16) (270, 300, 32) (270, 300, 60) (270, 300, 120)

mask_charge_O = tf.equal(chargesArray, 1)  # 创建形状为 (2, 300) 的布尔掩码
mask_charge_H = tf.equal(chargesArray, 2)  # 创建形状为 (2, 300) 的布尔掩码
for i in ["O", "H"]: 
        for j in ["O", "H"]:
            for k in ["A", "R"]:
                exec(f"Idx_{i}_{j}_{k} = []")
for m in range(pointsArray.shape[0]):  # 遍历批次
    for i in ["O", "H"]:
        exec(f"batch_mask_{i} = mask_charge_{i}[m]") # 当前批次掩码 
        for j in ["O", "H"]:
            for k in ["A", "R"]:
                exec(f"selected_{i}_{j}_{k} = tf.boolean_mask(Idx_{j}_{k}[m], batch_mask_{i})")
                exec(f"Idx_{i}_{j}_{k}.append(selected_{i}_{j}_{k})")
for i in ["O", "H"]:
    for j in ["O", "H"]:
        for k in ["A", "R"]:
            exec(f"Idx_{i}_{j}_{k} =  tf.stack(Idx_{i}_{j}_{k}, axis=0)")
            exec(f"print(Idx_{i}_{j}_{k}.shape)")


####   Test Set   ####
IdxTest_O_A, IdxTest_H_A, IdxTest_O_R, IdxTest_H_R = find_and_sort_neighbors_water(pointsTest, chargesTest, L, radious_A, maxNumNeighs_O_A, maxNumNeighs_H_A, radious_R, maxNumNeighs_O_R, maxNumNeighs_H_R)
mask_Test_O = tf.equal(chargesTest, 1)  # 创建形状为 (2, 300) 的布尔掩码
mask_Test_H = tf.equal(chargesTest, 2)  # 创建形状为 (2, 300) 的布尔掩码
for i in ["O", "H"]: 
        for j in ["O", "H"]:
            for k in ["A", "R"]:
                exec(f"IdxTest_{i}_{j}_{k} = []")
for m in range(pointsTest.shape[0]):  # 遍历批次
    for i in ["O", "H"]:
        exec(f"batchTest_mask_{i} = mask_Test_{i}[m]") # 当前批次掩码 
        for j in ["O", "H"]:
            for k in ["A", "R"]:
                exec(f"selectedTest_{i}_{j}_{k} = tf.boolean_mask(IdxTest_{j}_{k}[m], batchTest_mask_{i})")
                exec(f"IdxTest_{i}_{j}_{k}.append(selectedTest_{i}_{j}_{k})")
for i in ["O", "H"]:
    for j in ["O", "H"]:
        for k in ["A", "R"]:
            exec(f"IdxTest_{i}_{j}_{k} =  tf.stack(IdxTest_{i}_{j}_{k}, axis=0)")
            exec(f"print(IdxTest_{i}_{j}_{k}.shape)")

print("Training cycles in number of epochs")
print(Nepochs)
print("Training batch sizes for each cycle")
print(batchSizeArray)

errorlist_energy = []
errorlist_force = []
errorlist_energy_train = []
errorlist_force_train = []
losslist = []

### optimization parameters ###
mse_loss_fn = tf.keras.losses.MeanSquaredError()
initialLearningRate = learningRate
lrSchedule = tf.keras.optimizers.schedules.ExponentialDecay(
             initialLearningRate,
             decay_steps = ((Nsamples//10*9)//batchSizeArray[0])*epochsPerStair,
             decay_rate = decayRate,
             staircase = True)
optimizer = tf.keras.optimizers.Adam(learning_rate=lrSchedule)
loss_metric = tf.keras.metrics.Mean()

#############################################  load model  ########################################
if Test_type == "SLR2" or Load_short == 1:
  print("load Short model!")
  model.load_weights(model_load)
  model.compile()
  for layer in model.layers:
    print(f"{layer.name} is trainable: {layer.trainable}")

###################  training loop  ##################################
now = datetime.datetime.now()
time_name=now.strftime("%Y%m%d_%H%M%S")
min_test_F_err = np.Inf
min_test_E_err = np.Inf
for cycle, (epochs, batchSizeL) in enumerate(zip(Nepochs, batchSizeArray)):

  print('++++++++++++++++++++++++++++++', flush = True) 
  print('Start of cycle %d' % (cycle,))
  print('Total number of epochs in this cycle: %d'%(epochs,))
  print('Batch size in this cycle: %d'%(batchSizeL,))

  weightE = Energy_rate
  weightF = Force_rate

  x_train = (pointsArray, energyArray, forcesArray, chargesArray, Idx_O_O_A, Idx_H_O_A, Idx_O_H_A, Idx_H_H_A, Idx_O_O_R, Idx_H_O_R, Idx_O_H_R, Idx_H_H_R)
  train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
  train_dataset = train_dataset.shuffle(buffer_size=10000).batch(batchSizeL) # batchSizeL 2 4 8 16

  # Iterate over epochs
  for epoch in range(epochs):
    #start = time.time()
    #print('============================', flush = True) 
    #print('Start of epoch %d' % (epoch,))
  
    loss_metric.reset_state()
    
    #train_dataset = train_dataset.take(100) 
    #print(len(train_dataset))

    # Iterate over the batches of the dataset
    for step, x_batch_train in enumerate(train_dataset):
      start_epoch_time = time.time()

      #for i in range(14):
      #  print(i,"   ",x_batch_train[i].shape)
      
      #x_batch_train[0] 输入  x_batch_train[1] 能量输出  x_batch_train[2]力的输出
      loss = train_Water(model, optimizer, mse_loss_fn, x_batch_train[0], x_batch_train[3], L, select_neuron, inner_factor_A, radious_A, x_batch_train[4], x_batch_train[5], x_batch_train[6], x_batch_train[7], inner_factor_R, radious_R, x_batch_train[8], x_batch_train[9], x_batch_train[10], x_batch_train[11], Test_type, x_batch_train[1], x_batch_train[2], weightE, weightF)
      #end = time.time()
      #print('time 2 elapsed %.4f'%(end - start_epoch_time))

      loss_metric(loss)

    pottrain, forcetrain = model(pointsArray[:10], chargesArray[:10], tf.cast(LArray2, dtype = tf.float32), select_neuron, inner_factor_A, radious_A, Idx_O_O_A[:10], Idx_H_O_A[:10], Idx_O_H_A[:10], Idx_H_H_A[:10], inner_factor_R, radious_R, Idx_O_O_R[:10], Idx_H_O_R[:10], Idx_O_H_R[:10], Idx_H_H_R[:10], Test_type) 
    mae = tf.keras.losses.MeanAbsoluteError()
    err_train = mae(forcetrain, forcesArray[:10,:Npoints,:])
    #err_train = tf.sqrt(tf.reduce_mean(tf.square(forcetrain - forcesArray[:10,:Npoints,:])))
    
    #print(forcesArray[1,:,:])

    #errtrain = tf.sqrt(tf.reduce_sum(tf.square(forcetrain - forcesArray[:10,:,:])))/tf.sqrt(tf.reduce_sum(tf.square(forcetrain)))    

    #err_ener_train = tf.sqrt(tf.reduce_mean(tf.square(pottrain - energyArray[:10,:])))#/tf.sqrt(tf.reduce_sum(tf.square(pottrain)))
    err_ener_train = mae(pottrain, energyArray[:10])

    #print(pottrain.shape(),potentialArray.shape())
    #print("Relative Error in the trained energy is " +str(err_ener_train.numpy()))
    
    #print("Relative denom" + str(tf.sqrt(tf.reduce_sum(tf.square(forcetrain))).numpy()))
    #print("Relative Error in the trained energy is " +str(err_ener_train.numpy()))
    #print("Relative Error in the trained forces is " +str(errtrain.numpy()))

    potPred, forcePred = model(pointsTest, chargesTest, L, select_neuron,  inner_factor_A, radious_A, IdxTest_O_O_A, IdxTest_H_O_A, IdxTest_O_H_A, IdxTest_H_H_A, inner_factor_R, radious_R, IdxTest_O_O_R, IdxTest_H_O_R, IdxTest_O_H_R, IdxTest_H_H_R, Test_type)
    
    #err = tf.sqrt(tf.reduce_mean(tf.square(forcePred - forcesTest)))
    err = mae(forcePred, forcesTest)
    #err = tf.sqrt(tf.reduce_sum(tf.square(forcePred - forcesTest)))/tf.sqrt(tf.reduce_sum(tf.square(forcePred)))

    #err_ener = tf.sqrt(tf.reduce_mean(tf.square(potPred - energyTest)))#/tf.sqrt(tf.reduce_sum(tf.square(potPred)))
    err_ener = mae(potPred, energyTest) 

    if err < min_test_F_err:
      min_test_F_err = err
      #min_test_E_err = err_ener
      model.save_weights(saveFolder+time_name0+'_best_model.h5', save_format='h5')
      print("Relative Error in the trained forces is " +str(err_train.numpy()))
      #print("Relative Error in the trained energy is " +str(err_ener_train.numpy()))
      print("Relative Error in the test forces is " +str(err.numpy()))
      #print("Relative Error in the test energy is " +str(err_ener.numpy()))

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
  
  weights = model.layers[3].get_weights()  # 获取第1层的权重和偏置
  print(weights)
  #potPred, forcePred = model(pointsTest, chargesTest, neighListTest, L, radious)
  #err = mae(forcePred, forcesTest)
  #err_ener = mae(potPred, energyTest) 
  #print(err, err_ener)
  model.save_weights(saveFolder+time_name+'my_model_'+str_num+'.h5', save_format='h5')
  #model.load_weights(saveFolder+time_name+'my_model_'+str_num+'.h5')
  #weights = model.layers[3].get_weights()  # 获取第1层的权重和偏置
  #model.compile()
  #print(weights)
  #potPred, forcePred = model(pointsTest, chargesTest, neighListTest, L, radious)
  #err = mae(forcePred, forcesTest)
  #err_ener = mae(potPred, energyTest) 
  #print(err, err_ener)
  #model.save(saveFolder+time_name+'my_model_full_'+str_num+'.tf', save_format='tf')

