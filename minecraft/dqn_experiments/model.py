from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.layers import Embedding
import tensorflow as tf

class DQNetwork():
  def __init__(self, actions, input_shape,
               minibatch_size=None,
               learning_rate=0.000025,
               discount_factor=None, #9,
               dropout_prob=None,
               load_path=None,
               logger=None):

      # Parameters
      self.actions = actions 
      self.model = Sequential()

      self.model.add(Conv2D(32, 3, strides=(2, 2),
                            padding='valid',
                            activation='relu',
                            dtype=tf.float64, 
                            input_shape=input_shape,
                            data_format='channels_first'))

      # Third convolutional layer
      self.model.add(Conv2D(32, 2, strides=(1, 1),
                            padding='valid',
                            activation='relu',
                            dtype=tf.float64, #input_shape=input_shape,
                            data_format='channels_first'))


      # self.model.add(Embedding(3, 4,input_shape=[9,9]))
      # Second convolutional layer
      # self.model.add(Conv2D(32, 2, strides=(1, 1), 
      #                       padding='valid',
      #                       activation='relu',
      #                       input_shape=input_shape,
      #                       data_format='channels_first',dtype=tf.float64))
      # self.model.add(Conv2D(16, 2, strides=(1, 1), 
      #                       padding='valid',
      #                       activation='relu',dtype=tf.float64,
      #                      data_format='channels_first'))
      # # Third convolutional layer
      # self.model.add(Conv2D(4, 2, strides=(1, 1),
      #                       padding='valid',
      #                       activation='relu',
      #                       data_format='channels_first'))
      # self.model.add(Conv2D(4, 2, strides=(1, 1),
      #                       padding='valid',
      #                       activation='relu'))
#data_format='channels_first'))
      # Flatten the convolution output
      self.model.add(Flatten(dtype=tf.float64))

      # First dense layer
      self.model.add(Dense(128, activation='relu')) #512
      # self.model.add(Dense(32, input_shape=input_shape, activation='relu',dtype=tf.float64))
      # self.model.add(Dense(512, input_shape=input_shape, activation='relu'))

      # Output layer
      self.model.add(Dense(self.actions,dtype=tf.float64))

      # Load the network weights from saved model
      if load_path is not None:
          self.load(load_path)

      self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)#1e-3

      self.model.compile(loss='mean_squared_error',
                         optimizer='rmsprop',
                         metrics=['accuracy'])


  def loss(self,inputs, labels):
           return tf.keras.losses.mean_squared_error(inputs,labels)