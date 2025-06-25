import tensorflow as tf
import numpy as np 
import sys
from numba import jit 
from scipy.spatial.distance import cdist
import time
from scipy.spatial import cKDTree
from numba import vectorize, float32, njit

## Training functions ##
@tf.function
def train_pointcharge(model, optimizer, loss, inputs, neigh_list, charge, output_E, output_f, weight_e, weight_f):

  with tf.GradientTape() as tape:
    # we use the model the predict the outcome
    #train_time1 = time.time()
    predE, predF = model(inputs, charge, neigh_list, training=True)
    #predE, predF,_,_ = model(inputs, neigh_list, training=True)
    #train_time2 = time.time()
    #tf.print(f'Train loss took model {train_time2-train_time1:.6f} seconds')
    #tf.print(inputs[0,],neigh_list[0])
    #tf.print("inputs = :",type(input),type(neigh_list))
    #tf.print(predF[:,0,:],)
    #tf.print("predE = :", predE.shape,"  predF = :",predF.shape)

    # fidelity loss usin mse

    lossE = loss(predE, output_E)

    lossF = loss(predF, output_f)/output_f.shape[-1]
    
    #tf.print("lossE = :",lossE,"  LossF = :",lossF)

    total_loss = weight_e*lossE + weight_f*lossF
    #train_time3 = time.time()
    #tf.print(f'Compute loss took {train_time3-train_time2:.6f} seconds')
  # compute the gradients of the total loss with respect to the trainable variables
  #train_time4 = time.time()
  gradients = tape.gradient(total_loss, model.trainable_variables)


  #train_time5 = time.time()
  #tf.print(f'Update took {train_time5-train_time4:.6f} seconds')
  # update the parameters of the network
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  #train_time5 = time.time()
  #tf.print(f'Update took {train_time5-train_time4:.6f} seconds')

  return total_loss, gradients


# @tf.function
def train_dimer(model, optimizer, loss, inputs, neigh_list, charge, radious, output_E, output_f, weight_e, weight_f,Test_type):

  with tf.GradientTape() as tape:
    #print("Train here")
    #print(inputs.dtype)
    # we use the model the predict the outcome
    predE, predF = model(inputs, charge, neigh_list, radious, Test_type, training=True)
    # print(predE.dtype,output_E.dtype)
    # 找到真粒子的力用来做计算
    #print(predF.shape, output_f.shape)
    
    #tf.print("Here my shape",Lon_result.shape)
    lossE = loss(predE, output_E)
    # lossE = tf.reduce_mean(tf.square((predE - output_E)/output_E))
    lossF = loss(predF, output_f)/output_f.shape[-1]
    #tf.print(predE, output_E, (predE - output_E)/output_E)
    #tf.print("This is the train error of energy")
    #tf.print(lossE)
    #tf.print(tf.cast(predE, dtype = tf.float32) - tf.cast(output_E, dtype = tf.float32),summarize=-1)
    #tf.print("This is the inner")
    total_loss = weight_e * lossE + weight_f * lossF
    #print(total_loss.dtype)
    # 计算 L2 正则化项
    #l2_reg = tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables if 'bias' not in v.name])  # 忽略偏置项
    #l2_reg = 0.000001 * l2_reg  # lambda_l2 为正则化强度系数
    # 添加 L2 正则化到总损失
    #total_loss += tf.cast(l2_reg,dtype=tf.float32)

    #total_loss = weight_e*lossE + weight_f*lossF

  # compute the gradients of the total loss with respect to the trainable variables
  gradients = tape.gradient(total_loss, model.trainable_variables)
  '''
  for weight in model.trainable_weights:
    print(f"Weight: {weight.name}, dtype: {weight.dtype}")
  print("gradients dtype")
  for grad in gradients:
    if grad is not None:
        print(grad.dtype)
  '''
  # update the parameters of the network
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  return total_loss, gradients


@tf.function
def train_Water(model, optimizer, loss, inputs, charge, L, select_neuron, inner_factor_A, radious_A, Idx_O_O_A, Idx_H_O_A, Idx_O_H_A, Idx_H_H_A, inner_factor_R, radious_R, Idx_O_O_R, Idx_H_O_R, Idx_O_H_R, Idx_H_H_R, Test_type, output_E, output_f, weight_e, weight_f):
  with tf.GradientTape() as tape:
    # we use the model the predict the outcome
    predE, predF = model(inputs, charge, L, select_neuron, inner_factor_A, radious_A, Idx_O_O_A, Idx_H_O_A, Idx_O_H_A, Idx_H_H_A, inner_factor_R, radious_R, Idx_O_O_R, Idx_H_O_R, Idx_O_H_R, Idx_H_H_R, Test_type, training=True)
    #predE, predF = model(inputs, charge, neigh_list, training=True)
    lossE = loss(predE, output_E)
    lossF = loss(predF, output_f)/output_f.shape[-1]
      
    total_loss = weight_e * lossE + weight_f * lossF

  gradients = tape.gradient(total_loss, model.trainable_variables)

  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  #tf.print(total_loss, "total loss", lossE, lossF, predE, output_E)
  return total_loss #, gradients 