import tensorflow as tf
import time
from SOG_Net.lrc_layers import SOG_3DmixedSpecies_Water
from SOG_Net.utilities import pyramidLayer, DenseLayer
from SOG_Net.Neighbor import gen_coor_3d_species_water_Force_Double_List

class DPSOG_Water(tf.keras.Model):

  """Combines the encoder and decoder into an end-to-end model for training."""
  def __init__(self,
               Npoints,
               maxNumNeighs_O_A = 14,
               maxNumNeighs_H_A = 28,
               maxNumNeighs_O_R = 60,
               maxNumNeighs_H_R = 120,
               L = 14.4088,
               descripDim = [2, 4, 8, 16, 32],
               fittingDim = [16, 8, 4, 2, 1],
               NpointsFourier = 500, 
               fftChannels = 4,
               av = [0.0, 0.0],
               std = [1.0, 1.0],
               name='dpSOG_Water',
               **kwargs):
    super(DPSOG_Water, self).__init__(name=name, **kwargs)
    self.Npoints = Npoints
    # maximum number of neighbors
    self.maxNumNeighs_O_A = maxNumNeighs_O_A
    self.maxNumNeighs_H_A = maxNumNeighs_H_A
    self.maxNumNeighs_O_R = maxNumNeighs_O_R
    self.maxNumNeighs_H_R = maxNumNeighs_H_R
    # we normalize the inputs (should help for the training)
    self.av = av
    self.std = std
    self.descripDim = descripDim
    self.fittingDim = fittingDim
    self.descriptorDim = descripDim[-1]
    self.fftChannels = fftChannels

    # we may need to use the tanh here
    self.layerPyramid_O_O = pyramidLayer(descripDim, actfn = tf.nn.tanh) # GIA network
    self.layerPyramid_H_O = pyramidLayer(descripDim, actfn = tf.nn.tanh) # GIA network
    self.layerPyramid_O_H = pyramidLayer(descripDim, actfn = tf.nn.tanh) # GIA network
    self.layerPyramid_H_H = pyramidLayer(descripDim, actfn = tf.nn.tanh) # GIA network

    self.layerPyramidDir_O_O = pyramidLayer(descripDim, actfn = tf.nn.tanh) # GI network
    self.layerPyramidDir_H_O = pyramidLayer(descripDim, actfn = tf.nn.tanh) # GI network
    self.layerPyramidDir_O_H = pyramidLayer(descripDim, actfn = tf.nn.tanh) # GI network
    self.layerPyramidDir_H_H = pyramidLayer(descripDim, actfn = tf.nn.tanh) # GI network

    self.layerPyramid_Q = pyramidLayer([16, 16, 16], actfn = tf.nn.tanh)
    self.lrc_layer = SOG_3DmixedSpecies_Water(fftChannels, NpointsFourier, L)

    self.fittingNetwork = pyramidLayer(fittingDim, actfn = tf.nn.tanh) # Fit network
    self.linfitNet = DenseLayer(1)  # Ener network  
  
  @tf.function
  def call(self, inputs, charge_index, L, select_neuron, inner_factor_A, radious_A, Idx_O_O_A, Idx_H_O_A, Idx_O_H_A, Idx_H_H_A, inner_factor_R, radious_R, Idx_O_O_R, Idx_H_O_R, Idx_O_H_R, Idx_H_H_R, Test_type = "SR"):
    
    # we watch the inputs 
    with tf.GradientTape(persistent=True) as tape:
      tape.watch(inputs)  
      
      # For simulation
      RI_A, Sij_O_O_A, Sij_H_O_A, Sij_O_H_A, Sij_H_H_A, Sij_O_O_R, Sij_H_O_R, Sij_O_H_R, Sij_H_H_R = gen_coor_3d_species_water_Force_Double_List(inputs, L, inner_factor_A, radious_A, Idx_O_O_A, Idx_H_O_A, Idx_O_H_A, Idx_H_H_A, inner_factor_R, radious_R, Idx_O_O_R, Idx_H_O_R, Idx_O_H_R, Idx_H_H_R)
      
      # For Train
      #RI_A, Sij_O_O_A, Sij_H_O_A, Sij_O_H_A, Sij_H_H_A, Sij_O_O_R, Sij_H_O_R, Sij_O_H_R, Sij_H_H_R = find_and_sort_neighbors_water_Train(inputs, L, inner_factor_A, radious_A, Idx_O_O_A, Idx_H_O_A, Idx_O_H_A, Idx_H_H_A, inner_factor_R, radious_R, Idx_O_O_R, Idx_H_O_R, Idx_O_H_R, Idx_H_H_R)

      start_time = time.time() #计时
 
      #print(Sij_O_O_A.shape, Sij_H_O_A.shape, Sij_O_H_A.shape, Sij_H_H_A.shape) # (2, 100, 16, 1)
      # (10, 100, 25, 1) (10, 200, 25, 1) (10, 100, 50, 1) (10, 200, 50, 1)
      #print(Sij_O_O_R.shape, Sij_H_O_R.shape, Sij_O_H_R.shape, Sij_H_H_R.shape) # (2, 100, 16, 1)
      # (10, 100, 54, 1) (10, 200, 54, 1) (10, 100, 108, 1) (10, 200, 108, 1)

      GI_A_O_O = self.layerPyramid_O_O(Sij_O_O_A)
      GI_A_H_O = self.layerPyramid_H_O(Sij_H_O_A)
      GI_A_O_H = self.layerPyramid_O_H(Sij_O_H_A)
      GI_A_H_H = self.layerPyramid_H_H(Sij_H_H_A)

      #print(GI_A_O_O.shape, GI_A_H_O.shape, GI_A_O_H.shape, GI_A_H_H.shape)
      # (10, 100, 25, 100) (10, 200, 25, 100) (10, 100, 50, 100) (10, 200, 50, 100)
      
      # 沿第二个维度（axis=1）拼接
      GI_A_O = tf.concat([GI_A_O_O, GI_A_O_H], axis=2)
      GI_A_H = tf.concat([GI_A_H_O, GI_A_H_H], axis=2)
      GI_A = tf.concat([GI_A_O, GI_A_H], axis=1)

      #print(GI_A.shape, RI_A.shape)
      # (10, 300, 75, 100) (10, 300, 75, 4)
      #print(GI_A_O.shape, GI_A_H.shape)
      # 10, 100, 75, 100) (10, 200, 75, 100)
      
      RiaGia = tf.matmul(RI_A, GI_A, transpose_a=True) # (2, 300, 4, 100)
      #print(RiaGia.shape)
      # (10, 300, 4, 100)
      end_time = time.time() #计时
      #print(f'Step Descriptor took {end_time - start_time:.6f} seconds')
      start_time = time.time() #计时
     
      #print(select_neuron, ((self.maxNumNeighs_O_A+self.maxNumNeighs_H_A)**2))
      # 10 5625

      # 取前select_neuron个维度
      Da = tf.matmul(RiaGia[:,:,:,:select_neuron], RiaGia, transpose_a=True)/((self.maxNumNeighs_O_A+self.maxNumNeighs_H_A)**2) # (2, 25, 4, 100)
      #print(Da.shape, ((self.maxNumNeighs_O_A+self.maxNumNeighs_H_A)**2)) # (4, 300, 100, 100) 
      # (10, 300, 10, 100) 5625

      GI_R_O_O = self.layerPyramidDir_O_O(Sij_O_O_R)
      GI_R_H_O = self.layerPyramidDir_H_O(Sij_H_O_R)
      GI_R_O_H = self.layerPyramidDir_O_H(Sij_O_H_R)
      GI_R_H_H = self.layerPyramidDir_H_H(Sij_H_H_R)
      
      #tf.print(Sij_O_O_R[0,0,:,0],GI_R_O_O[0,0,:,0])
      #tf.print(GI_R_O_O.shape, GI_R_H_O.shape, GI_R_O_H.shape, GI_R_H_H.shape)
      # (30, 100, 60, 100) (30, 200, 60, 100) (30, 100, 120, 100) (30, 200, 120, 100)

      # 沿第二个维度（axis=1）拼接
      GI_R_O = tf.concat([GI_R_O_O, GI_R_O_H], axis=2)
      GI_R_H = tf.concat([GI_R_H_O, GI_R_H_H], axis=2)
      GI_R = tf.concat([GI_R_O, GI_R_H], axis=1)
      
      #print(GI_R_O.shape, GI_R_H.shape, GI_R.shape)
      # (10, 100, 162, 100) (10, 200, 162, 100) (10, 300, 162, 100)
      
      Dr = tf.reduce_mean(GI_R, axis=2) # (2, 25, 100)
      #print(Dr.shape)
      
      D = tf.concat([tf.reshape(Da, (Da.shape[0], Da.shape[1], Da.shape[2] * Da.shape[3])), Dr], axis=2) # (2, 25, 500)
      #print(D.shape)

      Fit = self.fittingNetwork(D) # (2, 25, 64)
      Ea = self.linfitNet(Fit) # (2, 25, 1)
      Ea = tf.squeeze(Ea, axis=[2])
      Ea = tf.reduce_sum(Ea, axis=[-1])
      
      end_time = time.time() #计时
      #print(f'Step Fitting took {end_time - start_time:.6f} seconds')
      start_time = time.time() #计时

      # 生成电荷Q
      # Q = self.layerPyramid_Q(Da)
      # 生成电荷Q
      
      #if Test_type!="SR":
      Q = self.layerPyramid_Q(D)
      
      end_time = time.time() #计时
      #print(f'Step Gene Q took {end_time - start_time:.6f} seconds')
      start_time = time.time() #计时

      #if Test_type!="SR":
      
      long_range_coord, Force_long = self.lrc_layer(inputs, Q) # (2, 25, 5, 16)
      El = tf.reduce_sum(long_range_coord, axis=[-1])

      #print(long_range_coord.shape)
      #Forces = -tape.gradient(El, inputs)
      #tf.print(Forces[0,1:5,:],Forces.shape)
      #tf.print(Force_long[0,1:5,:],Force_long.shape)
      #else:
      #    long_range_coord = self.lrc_layer(inputs[0:2,...], Q[0:2,...]) # (2, 25, 5, 16)

      end_time = time.time() #计时
      #print(f'Step Long-range took {end_time - start_time:.6f} seconds')

      if Test_type=="SR":
        Energy = Ea
      elif Test_type=="LR":  
        Energy = El
      else:
        Energy = Ea + El
    
    if Test_type=="SR":
      Forces = -tape.gradient(Ea, inputs)
    elif Test_type=="LR":
      Forces = Force_long
    else:
      Forces = -tape.gradient(Ea, inputs) + Force_long
    
    #tf.print(Forces[0,1:5,:])
    #Forces = -tape.gradient(Energy, inputs)
    #tf.print(Forces[0,1:5,:])
    #tf.print(Energy,GI_A,"Energy")
    return Energy, Forces 