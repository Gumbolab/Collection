#create new cell
class fo_poolcell(tf.nn.rnn_cell.RNNCell):
          def __init__(self,fo_out):
                  self.__fo_out=fo_out
                  
          @property
          def output_size(self):
            return self.__fo_out
          @property 
          def state_size(self):
            return self.__fo_out
          def __call__(self,inputs,state):
                    """Run this RNN cell on inputs, starting from the given state.
                          Args:
                            inputs: `2-D` tensor with shape `[batch_size, input_size]`.but in fo_pool inputs shape[batch_size,input_state*3]
                            state: if `self.state_size` is an integer, this should be a `2-D Tensor`
                              with shape `[batch_size, self.state_size]`.
                          
                          Returns:
                            A pair containing:
                            - Output: A `2-D` tensor with shape `[batch_size, self.output_size]`.
                            - New state: Either a single `2-D` tensor, or a tuple of tensors matching
                              the arity and shapes of `state`.
                          """
                    Z,F,O=tf.split(inputs,3,1)
                    with tf.variable_scope("fo_cell"):
                          new_state=tf.multiply(F,state)+tf.multiply(tf.subtract(1.,F),Z)
                          outputs=tf.multiply(new_state, O)
                    return outputs,new_state
class Quasi_layer:
          
          def __init__(self,fwidth,out_Qmap):#feed in k, m
                  self.__fwidth=fwidth
                  self.__out_Qmap=out_Qmap
          def __call__(self,inputs):
                  """
                  inputs is a tensor have shape=[batch_size,max_time, in_shape]
                  outputs return after convolution layer shape=[bacth_size,max_time,filter_width*3]
                      """  
                  _out_Qmap=self.__out_Qmap
                  _fwidth=self.__fwidth
                  batch_size=inputs.get_shape().as_list()[0]
                  with tf.variable_scope ("Qrnn_layer"):
                            out_conv=self.convolution(inputs,filter_width,out_cmaps)
                            ##create pool layer, create cell_pool like rnn sequences
                            fo_fw=FopoolCell(_out_Qmap)
                            fo_bw=FopoolCell(_out_Qmap)
                            
                            initcell_fw=cell_fw.zero_state(batch_size,tf.float32)
                            initcell_bw=cell_bw.zero_state(batch_size,tf.float32)
                            (output_fw, output_bw), last_state=tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,cell_bw=cell_bw,inputs=inputs,initial_state_fw=initcell_fw,initial_state_bw=initcell_bw)
                            outputs = tf.concat([output_fw, output_bw], axis=2)
                            return outputs
                                    

          def convolution(self,_inputs,_fwidth,_out_Qmap):
                """
                inputs: data series shape[batch_size,max_time,num_state]
                convolution subcomponent: inputs should be padding to have the same max_time in out_put
                output: Z,F,O in vector shape[batch_size,max_time, out_num_state],out_num_state equal num_of filter 
                filters shape [k,n,m],k: width of filter,n: num_state inputs, m: numbers of filter
                define filter as variable in tensorflow
                """
                in_Qmap=_inputs.get_shape().as_list()[-1]
                _inputs_pad=tf.pad(_inputs,[[0,0],[0,_fwidth-1]])
                with tf.variable_scope("convolution_scope"):
                              """
                              wz,wf,wo: init uniform shape [fwidth,num_input,out_Qmap]
                              Z = tanh(Wz ∗ X)
                              F = σ(Wf ∗ X)
                              O = σ(Wo ∗ X),
                            
                              """
                              wz=tf.get_variable(name="wz",shape=[_fwidth,in_Qmap,_out_Qmap],initializer=tf.random_uniform_initializer(minval=-.05, maxval=.05))
                              wf=tf.get_variable(name="wf",shape=[_fwidth,in_Qmap,_out_Qmap],initializer=tf.random_uniform_initializer(minval=-.05, maxval=.05))
                              wo=tf.get_variable(name="wo",shape=[_fwidth,in_Qmap,_out_Qmap],initializer=tf.random_uniform_initializer(minval=-.05, maxval=.05))
                              #compute gate Z,F,O
                              z_a = tf.nn.conv1d(pad_input, wz, stride=1, padding='VALID')
                              Z=tf.tanh(z_a)
                              f_a = tf.nn.conv1d(pad_input, wf, stride=1, padding='VALID')+bf
                              F=tf.sigmoid(f_a)
                              #O gate
                            
                              o_a = tf.nn.conv1d(pad_input, wo, stride=1, padding='VALID')+bo
                              O=tf.sigmoid(o_a)
                              outputs=tf.concat([Z,F,O],2)
                return outputs

