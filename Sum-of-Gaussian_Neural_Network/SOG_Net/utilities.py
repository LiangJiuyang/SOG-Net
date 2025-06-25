import tensorflow as tf
import numpy as np 
import sys
from numba import jit 
from scipy.spatial.distance import cdist
import time
from scipy.spatial import cKDTree
from numba import vectorize, float32, njit

class DenseLayer(tf.keras.layers.Layer):
  def __init__(self, num_outputs):
    super(DenseLayer, self).__init__()
    self.num_outputs = num_outputs

  def build(self, input_shape):
    self.kernel = self.add_weight(name="kernel",
                                  initializer=tf.initializers.GlorotNormal(),
                                  shape=[int(input_shape[-1]),
                                         self.num_outputs])
    self.bias = self.add_weight(name="bias",
                                initializer=tf.initializers.GlorotNormal(),    
                                shape=[self.num_outputs,])
  @tf.function
  def call(self, input):
    return tf.matmul(input, self.kernel) + self.bias


class pyramidLayer(tf.keras.layers.Layer):
  def __init__(self, num_outputs, actfn = tf.nn.relu):
    super(pyramidLayer, self).__init__()
    self.num_outputs = num_outputs
    self.actfn = actfn

  def build(self, input_shape):
    self.kernel = []
    self.bias = []
    #print(int(input_shape[-1]))
    #print(self.num_outputs[0])
    self.kernel.append(self.add_weight(name="kernel",
                       initializer=tf.initializers.GlorotNormal(),
                       shape=[int(input_shape[-1]),self.num_outputs[0]]))
    
    self.bias.append(self.add_weight(name="bias",
                       initializer=tf.initializers.GlorotNormal(),
                       shape=[self.num_outputs[0],]))

    for n, (l,k) in enumerate(zip(self.num_outputs[0:-1],  
                                  self.num_outputs[1:])) :

      self.kernel.append(self.add_weight(name="kernel"+str(n),
                         shape=[l, k]))
      self.bias.append(self.add_weight(name="bias"+str(n),
                         shape=[k,]))

  @tf.function
  def call(self, input):
    x = self.actfn(tf.matmul(input, self.kernel[0]) + self.bias[0])
    for k, (ker, b) in enumerate(zip(self.kernel[1:], self.bias[1:])):
      if self.num_outputs[k] == self.num_outputs[k+1]:
        x += self.actfn(tf.matmul(x, ker) + b)  ###ResNet
      else :
        x = self.actfn(tf.matmul(x, ker) + b)
    return x

class pyramidLayer_ThreeBodyQ_DNN(tf.keras.layers.Layer):
  def __init__(self, num_outputs, actfn = tf.nn.relu):
    super(pyramidLayer_ThreeBodyQ_DNN, self).__init__()
    self.num_outputs = num_outputs
    self.actfn = actfn

  def build(self, input_shape):
    self.kernel = []
    self.bias = []
    #print(int(input_shape[-1]))
    #print(self.num_outputs[0])
    self.kernel.append(self.add_weight(name="kernel",
                       initializer=tf.initializers.GlorotNormal(),
                       shape=[int(input_shape[-1]),self.num_outputs[0]],dtype=tf.float32))
    
    self.bias.append(self.add_weight(name="bias",
                       initializer=tf.initializers.GlorotNormal(),
                       shape=[self.num_outputs[0],],dtype=tf.float32))

    for n, (l,k) in enumerate(zip(self.num_outputs[0:-1],  
                                  self.num_outputs[1:])) :

      self.kernel.append(self.add_weight(name="kernel"+str(n),
                         initializer=tf.initializers.GlorotNormal(),
                         shape=[l, k],dtype=tf.float32))
      self.bias.append(self.add_weight(name="bias"+str(n),
                         initializer=tf.initializers.GlorotNormal(),
                         shape=[k,],dtype=tf.float32))

  @tf.function
  def call(self, input):
    input = tf.cast(input, tf.float32)
    x = self.actfn(tf.matmul(input, self.kernel[0]) + self.bias[0])
    for k, (ker, b) in enumerate(zip(self.kernel[1:], self.bias[1:])):
      if self.num_outputs[k] == self.num_outputs[k+1]:
        x += self.actfn(tf.matmul(x, ker) + b)  ###ResNet
      else :
        x = self.actfn(tf.matmul(x, ker) + b)
    return x

@tf.function
def gen_coor_3d(r_in, neigh_list, L, 
                            av = tf.constant([0.0, 0.0], dtype = tf.float32),
                            std =  tf.constant([1.0, 1.0], dtype = tf.float32)):

    # This function follows the same trick 
    # function to generate the generalized coordinates for periodic data
    # neigh_list is a (Nsample, Npoints, maxNeigh)

    Nsamples = r_in.shape[0]
    max_num_neighs = neigh_list.shape[-1]


    # define an indicator
    mask = neigh_list > -1

    # extract per_dimension the repeated and gathered entries
    r_in_rep_X  = tf.tile(tf.expand_dims(r_in[:,:,0], -1),
                       [1 ,1, max_num_neighs] )   
    r_in_gath_X = tf.gather(r_in[:,:,0], neigh_list, 
                           batch_dims = 1, axis = 1)
    r_in_rep_Y  = tf.tile(tf.expand_dims(r_in[:,:,1], -1),
                       [1 ,1, max_num_neighs] )
    r_in_gath_Y = tf.gather(r_in[:,:,1], neigh_list, 
                           batch_dims = 1, axis = 1)
    r_in_rep_Z  = tf.tile(tf.expand_dims(r_in[:,:,2], -1),
                       [1 ,1, max_num_neighs] )
    r_in_gath_Z = tf.gather(r_in[:,:,2], neigh_list, 
                           batch_dims = 1, axis = 1)


    # compute the periodic dimension wise distance
    r_diff_X = r_in_gath_X - r_in_rep_X
    r_diff_X = r_diff_X - L*tf.round(r_diff_X/L)
    r_diff_Y = r_in_gath_Y - r_in_rep_Y
    r_diff_Y = r_diff_Y - L*tf.round(r_diff_Y/L)
    r_diff_Z = r_in_gath_Z - r_in_rep_Z
    r_diff_Z = r_diff_Z - L*tf.round(r_diff_Z/L)

    norm = tf.sqrt(tf.square(r_diff_X) + tf.square(r_diff_Y) + tf.square(r_diff_Z))

    binv = tf.math.reciprocal(norm) 
    bx = tf.math.multiply(r_diff_X, binv)
    by = tf.math.multiply(r_diff_Y, binv)
    bz = tf.math.multiply(r_diff_Z, binv)

    zeroDummy = tf.zeros_like(norm)
    # add zero when the actual number of neighbors are less than maxNumNeigh
    binv_safe = tf.where(mask, (binv- av[0])/std[0], zeroDummy)
    bx_safe = tf.where(mask, bx, zeroDummy)
    by_safe = tf.where(mask, by, zeroDummy)
    bz_safe = tf.where(mask, bz, zeroDummy)
    
    r_total = tf.concat([tf.reshape(binv_safe, (-1,1)), 
                         tf.reshape(bx_safe,   (-1,1)), 
                         tf.reshape(by_safe,   (-1,1)),
                         tf.reshape(bz_safe,   (-1,1)) ], axis = 1)

    return r_total

@tf.function
def gen_coor_3d_Q(r_in,charge, neigh_list, L, av = tf.constant([0.0, 0.0], dtype = tf.float32),std =  tf.constant([1.0, 1.0], dtype = tf.float32)):

    # This function follows the same trick 
    # function to generate the generalized coordinates for periodic data
    # neigh_list is a (Nsample, Npoints, maxNeigh)

    Nsamples = r_in.shape[0]
    max_num_neighs = neigh_list.shape[-1]

    # define an indicator
    mask = neigh_list > -1

    # extract per_dimension the repeated and gathered entries
    r_in_rep_X  = tf.tile(tf.expand_dims(r_in[:,:,0], -1), [1 ,1, max_num_neighs])
    r_in_gath_X = tf.gather(r_in[:,:,0], neigh_list, batch_dims = 1, axis = 1)

    r_in_rep_Y  = tf.tile(tf.expand_dims(r_in[:,:,1], -1), [1 ,1, max_num_neighs] )
    r_in_gath_Y = tf.gather(r_in[:,:,1], neigh_list, batch_dims = 1, axis = 1)
    
    r_in_rep_Z  = tf.tile(tf.expand_dims(r_in[:,:,2], -1), [1 ,1, max_num_neighs] )
    r_in_gath_Z = tf.gather(r_in[:,:,2], neigh_list, batch_dims = 1, axis = 1)

    #print(r_in.shape)
    #print(r_in[:,:,0].shape)
    #print(charge.shape)

    C_in_rep  = tf.tile(tf.expand_dims(charge, -1), [1 ,1, max_num_neighs])

    C_in_rep = tf.cast(C_in_rep, dtype=tf.float32)

    C_in_gath = tf.gather(charge, neigh_list, batch_dims = 1, axis = 1)

    C_in_gath = tf.cast(C_in_gath, dtype=tf.float32)

    #print(C_in_gath.dtype)
    #tf.print(C_in_gath[1,1,1])
    #tf.print(C_in_rep.shape,C_in_gath.shape)
    #tf.print(charge.shape,neigh_list.shape)
    

    # compute the periodic dimension wise distance
    r_diff_X = r_in_gath_X - r_in_rep_X
    r_diff_X = r_diff_X - L*tf.round(r_diff_X/L)
    r_diff_Y = r_in_gath_Y - r_in_rep_Y
    r_diff_Y = r_diff_Y - L*tf.round(r_diff_Y/L)
    r_diff_Z = r_in_gath_Z - r_in_rep_Z
    r_diff_Z = r_diff_Z - L*tf.round(r_diff_Z/L)

    norm = tf.sqrt(tf.square(r_diff_X) + tf.square(r_diff_Y) + tf.square(r_diff_Z))
   
    #tf.print(norm.shape)
    #time.sleep(4)

    binv = tf.math.reciprocal(norm) 
    bx = tf.math.multiply(r_diff_X, binv)
    by = tf.math.multiply(r_diff_Y, binv)
    bz = tf.math.multiply(r_diff_Z, binv)

    zeroDummy = tf.zeros_like(norm)
    # add zero when the actual number of neighbors are less than maxNumNeigh
    binv_safe = tf.where(mask, (binv- av[0])/std[0], zeroDummy)
    bx_safe = tf.where(mask, bx, zeroDummy)
    by_safe = tf.where(mask, by, zeroDummy)
    bz_safe = tf.where(mask, bz, zeroDummy)

    binv_Q = C_in_rep * C_in_gath * binv
    bx_Q = tf.math.multiply(r_diff_X, binv_Q)
    by_Q = tf.math.multiply(r_diff_Y, binv_Q)
    bz_Q = tf.math.multiply(r_diff_Z, binv_Q)

    binv_safe_Q = tf.where(mask, (binv_Q - av[1])/std[1], zeroDummy)
    bx_safe_Q = tf.where(mask, bx_Q, zeroDummy)
    by_safe_Q = tf.where(mask, by_Q, zeroDummy)
    bz_safe_Q = tf.where(mask, bz_Q, zeroDummy)

    #print("Hi")
    # print(binv_safe.shape, bx_safe_Q.shape)
    #print((C_in_gath*binv_safe).shape)
    #print((C_in_rep*C_in_gath*binv_safe).shape)
    #print((C_in_gath*binv_safe).dtype)
    
    # 将三个张量沿着新轴拼接成一个形状为 100x100x100x3 的张量
    #stacked_tensor = tf.stack([binv_safe, bx_safe, by_safe, bz_safe, binv_safe_Q, bx_safe_Q, by_safe_Q, bz_safe_Q], axis=-1)

    #print(stacked_tensor.shape)

    r_total = tf.concat([tf.reshape(binv_safe, (-1,1)), 
                          tf.reshape(bx_safe,   (-1,1)), 
                          tf.reshape(by_safe,   (-1,1)),
                          tf.reshape(bz_safe,   (-1,1)),
                          tf.reshape(binv_safe_Q, (-1,1)),
                          tf.reshape(bx_safe_Q,   (-1,1)), 
                          tf.reshape(by_safe_Q,   (-1,1)),
                          tf.reshape(bz_safe_Q,   (-1,1))  ], axis = 1)
    
    #tf.print("shape ", binv_safe.shape, bx_safe.shape)
    return r_total #stacked_tensor #r_total

###  近邻查找  ###
def find_and_sort_neighbors_dimer(Rinnumpy, chargesArray, L, radious, maxNumNeighs):
    n_samples, n_points, dimension = Rinnumpy.shape
    #print(n_samples, n_points)
    Idx = np.zeros((n_samples, n_points, maxNumNeighs), dtype=np.int32)-1
    for i in range(n_samples):
      r_sample = Rinnumpy[i]
      c_sample = chargesArray[i]
      tree = cKDTree(r_sample, boxsize=[L, L, L])
      r_list = tree.query_ball_point(r_sample,radious)
      #for j,row in enumerate(r_list):
      #   print(chargesArray[i,row])
      r_list = [[elem for elem in row if elem!=i and c_sample[elem] != 0] for i,row in enumerate(r_list)] 
      #print("Ordered neighbor list")
      for j, row in enumerate(r_list):
        # 排序 但没考虑周期边界条件
        if len(row) > 0:
           # 计算到每个邻居的距离
           distances = cdist([r_sample[j]], r_sample[row]).flatten()
           # 按照距离排序
           sorted_indices = np.argsort(distances)
           sorted_neighbors = np.array(row)[sorted_indices]
           # 将排序后的邻居存入 Idx
           Idx[i, j, :len(sorted_neighbors)] = sorted_neighbors
        # 不排序
        #Idx[i,j,:len(row)]=row
        #print(chargesArray[i,row])

    return Idx

def find_and_sort_neighbors_water(Rinnumpy, chargesArray, L, radious_A, maxNumNeighs_O_A, maxNumNeighs_H_A, radious_R, maxNumNeighs_O_R, maxNumNeighs_H_R):
    n_samples, n_points, dimension = Rinnumpy.shape

    #print(Rinnumpy.shape,chargesArray.shape,L,"her6")

    Idx_O_A = np.zeros((n_samples, n_points, maxNumNeighs_O_A), dtype=np.int32) - 1
    Idx_H_A = np.zeros((n_samples, n_points, maxNumNeighs_H_A), dtype=np.int32) - 1
    Idx_O_R = np.zeros((n_samples, n_points, maxNumNeighs_O_R), dtype=np.int32) - 1
    Idx_H_R = np.zeros((n_samples, n_points, maxNumNeighs_H_R), dtype=np.int32) - 1
    
    #print(Rinnumpy.shape,chargesArray.shape, chargesArray[0])

    for i in range(n_samples):
      #print(i)
      r_sample = Rinnumpy[i]
      c_sample = chargesArray[i]
      #print(c_sample.shape,r_sample.shape)
      tree = cKDTree(r_sample, boxsize=[L, L, L])
      r_list_A = tree.query_ball_point(r_sample, radious_A)
      r_list_R = tree.query_ball_point(r_sample, radious_R)
      #for j,row in enumerate(r_list):
      #   print(chargesArray[i,row])
      #print(r_list_A) 
      r_list_O_A = [[elem for elem in row if elem!=j and c_sample[elem] == 1] for j,row in enumerate(r_list_A)] 
      r_list_H_A = [[elem for elem in row if elem!=j and c_sample[elem] == 2] for j,row in enumerate(r_list_A)] 
      r_list_O_R = [[elem for elem in row if elem!=j and c_sample[elem] == 1] for j,row in enumerate(r_list_R)]
      r_list_H_R = [[elem for elem in row if elem!=j and c_sample[elem] == 2] for j,row in enumerate(r_list_R)]
      
      #print(r_list_O_A[0],r_list_H_A[0],r_list_O_R[0])

      for j,row in enumerate(r_list_O_A):
         # 不排序
         if len(row)<=maxNumNeighs_O_A:
            Idx_O_A[i,j,:len(row)]=row
         else:
            print("Exceed Maximum maxNumNeighs_O_A")
            Idx_O_A[i,j,:maxNumNeighs_O_A]=row[:maxNumNeighs_O_A]
      
      for j,row in enumerate(r_list_H_A):
         # 不排序
         if len(row)<=maxNumNeighs_H_A:
            Idx_H_A[i,j,:len(row)]=row
         else:
            print("Exceed Maximum maxNumNeighs_H_A")
            Idx_H_A[i,j,:maxNumNeighs_H_A]=row[:maxNumNeighs_H_A]

      for j,row in enumerate(r_list_O_R):
         # 不排序
         if len(row)<=maxNumNeighs_O_R:
            Idx_O_R[i,j,:len(row)]=row
         else:
            print("Exceed Maximum maxNumNeighs_O_R")
            Idx_O_R[i,j,:maxNumNeighs_O_R]=row[:maxNumNeighs_O_R]
      
      for j,row in enumerate(r_list_H_R):
         # 不排序
         if len(row)<=maxNumNeighs_H_R:
            Idx_H_R[i,j,:len(row)]=row
         else:
            print("Exceed Maximum maxNumNeighs_H_R")
            Idx_H_R[i,j,:maxNumNeighs_H_R]=row[:maxNumNeighs_H_R]

    return Idx_O_A, Idx_H_A, Idx_O_R, Idx_H_R