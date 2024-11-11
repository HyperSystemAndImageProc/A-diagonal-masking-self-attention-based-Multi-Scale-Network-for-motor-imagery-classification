""" 
Copyright (C) 2022 King Saud University, Saudi Arabia 
SPDX-License-Identifier: Apache-2.0 

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the 
License at

http://www.apache.org/licenses/LICENSE-2.0  

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR 
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. 

Author:  Hamdi Altaheri 
"""

#%%
import math
import tensorflow as tf
from keras.activations import softmax
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense,BatchNormalization,GlobalAveragePooling1D,GlobalMaxPooling1D
from tensorflow.keras.layers import multiply, Permute, Concatenate, Conv2D, Add, Activation, Lambda,Conv1D
from tensorflow.keras.layers import Dropout, MultiHeadAttention, LayerNormalization, Reshape
from tensorflow.keras import backend as K
from tensorflow.keras import layers

#%% Create and apply the attention model
def attention_block(net, attention_model):
	in_sh = net.shape # dimensions of the input tensor
	in_len = len(in_sh)
	expanded_axis = 3 # defualt = 3

	if attention_model == 'mha':   # Multi-head self attention layer
		if(in_len > 3):
			net = Reshape((in_sh[1],-1))(net)
		net = mha_block(net)
	elif attention_model == 'mhla':  # Multi-head local self-attention layer
		if(in_len > 3):
			net = Reshape((in_sh[1],-1))(net)
		net = mha_block(net, vanilla = False)
	elif attention_model == 'se':   # Squeeze-and-excitation layer
		if(in_len < 4):
			net = tf.expand_dims(net, axis=expanded_axis)
		net = se_block(net, ratio=8)
	elif attention_model == 'cbam': # Convolutional block attention module
		if(in_len < 4):
			net = tf.expand_dims(net, axis=expanded_axis)
		net = cbam_block(net, ratio=8)
	else:
		raise Exception("'{}' is not supported attention module!".format(attention_model))

	if (in_len == 3 and len(net.shape) == 4):
		net = K.squeeze(net, expanded_axis)
	elif (in_len == 4 and len(net.shape) == 3):
		net = Reshape((in_sh[1], in_sh[2], in_sh[3]))(net)
	return net


#%% Multi-head self Attention (MHA) block
def mha_block(input_feature, key_dim=8, num_heads=2, dropout = 0.5, vanilla = True):
	"""Multi Head self Attention (MHA) block.

	Here we include two types of MHA blocks:
			The original multi-head self-attention as described in https://arxiv.org/abs/1706.03762
			The multi-head local self attention as described in https://arxiv.org/abs/2112.13492v1
	"""
	# Layer normalization
	x = LayerNormalization(epsilon=1e-6)(input_feature)

	if vanilla:
		# Create a multi-head attention layer as described in
		# 'Attention Is All You Need' https://arxiv.org/abs/1706.03762
		x = MultiHeadAttention(key_dim = key_dim, num_heads = num_heads, dropout = dropout)(x, x)
	else:
		# Create a multi-head local self-attention layer as described in
		# 'Vision Transformer for Small-Size Datasets' https://arxiv.org/abs/2112.13492v1

		# Build the diagonal attention mask
		NUM_PATCHES = input_feature.shape[1]
		diag_attn_mask = 1 - tf.eye(NUM_PATCHES)
		diag_attn_mask = tf.cast([diag_attn_mask], dtype=tf.int8)

		# Create a multi-head local self attention layer.
		x = MultiHeadAttention_LSA(key_dim = key_dim, num_heads = num_heads, dropout = dropout)(
			x, x, attention_mask = diag_attn_mask)

	x = Dropout(0.3)(x)
	# Skip connection
	mha_feature = Add()([input_feature, x])

	return mha_feature


def EANet(input_feature):
	x = LayerNormalization(epsilon=1e-6)(input_feature)
	x=external_attention(x,dim=8,num_heads=2)
	ea_feature = Add()([input_feature, x])
	return ea_feature
def external_attention(
		x, dim, num_heads, dim_coefficient=4, attention_dropout=0.5, projection_dropout=0.3
):
	_, num_patch, channel = x.shape
	assert dim % num_heads == 0
	num_heads = num_heads * dim_coefficient

	x = layers.Dense(dim * dim_coefficient)(x)
	# create tensor [batch_size, num_patches, num_heads, dim*dim_coefficient//num_heads]
	x = tf.reshape(
		x, shape=(-1, num_patch, num_heads, dim * dim_coefficient // num_heads)
	)
	x = tf.transpose(x, perm=[0, 2, 1, 3])
	# a linear layer M_k
	attn = layers.Dense(dim // dim_coefficient)(x)
	# normalize attention map
	attn = layers.Softmax(axis=2)(attn)
	# dobule-normalization
	attn = attn / (1e-9 + tf.reduce_sum(attn, axis=-1, keepdims=True))
	attn = layers.Dropout(attention_dropout)(attn)
	# a linear layer M_v
	x = layers.Dense(dim * dim_coefficient // num_heads)(attn)
	x = tf.transpose(x, perm=[0, 2, 1, 3])
	x = tf.reshape(x, [-1, num_patch, dim * dim_coefficient])
	# a linear layer to project original dim
	x = layers.Dense(32)(x)
	x = layers.Dropout(projection_dropout)(x)
	return x


#%% Multi head self Attention (MHA) block: Locality Self Attention (LSA)
class MultiHeadAttention_LSA(tf.keras.layers.MultiHeadAttention):
	"""local multi-head self attention block

	 Locality Self Attention as described in https://arxiv.org/abs/2112.13492v1
	 This implementation is taken from  https://keras.io/examples/vision/vit_small_ds/
	"""
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		# The trainable temperature term. The initial value is the square
		# root of the key dimension.
		self.tau = tf.Variable(math.sqrt(float(self._key_dim)), trainable=True)

	def _compute_attention(self, query, key, value, attention_mask=None, training=None):
		query = tf.multiply(query, 1.0 / self.tau)
		attention_scores = tf.einsum(self._dot_product_equation, key, query)
		attention_scores = self._masked_softmax(attention_scores, attention_mask)
		attention_scores_dropout = self._dropout_layer(
			attention_scores, training=training
		)
		attention_output = tf.einsum(
			self._combine_equation, attention_scores_dropout, value
		)
		return attention_output, attention_scores


#%% Squeeze-and-excitation block
def se_block(input_feature, ratio=8):
	"""Squeeze-and-Excitation(SE) block.

	As described in https://arxiv.org/abs/1709.01507
	The implementation is taken from https://github.com/kobiso/CBAM-keras
	"""
	channel_axis = 1 if K.image_data_format() == "channels_first" else -1
	channel = input_feature.shape[channel_axis]

	se_feature = GlobalAveragePooling2D()(input_feature)
	se_feature = Reshape((1, 1, channel))(se_feature)
	assert se_feature.shape[1:] == (1,1,channel)
	se_feature = Dense(channel // ratio,
					   activation='elu',
					   kernel_initializer='he_normal',
					   use_bias=True,
					   bias_initializer='zeros')(se_feature)
	assert se_feature.shape[1:] == (1,1,channel//ratio)
	se_feature = Dense(channel,
					   activation='sigmoid',
					   kernel_initializer='he_normal',
					   use_bias=True,
					   bias_initializer='zeros')(se_feature)
	assert se_feature.shape[1:] == (1,1,channel)
	if K.image_data_format() == 'channels_first':
		se_feature = Permute((3, 1, 2))(se_feature)

	se_feature = multiply([input_feature, se_feature])
	return se_feature

def cse_block(input_feature, ratio=2):
	"""Squeeze-and-Excitation(SE) block.

	As described in https://arxiv.org/abs/1709.01507
	The implementation is taken from https://github.com/kobiso/CBAM-keras
	"""
	# channel_axis = 1 if K.image_data_format() == "channels_first" else -1
	# channel = input_feature.shape[channel_axis]
	channel_axis = 1 if K.image_data_format() == "channels_first" else -1
	channel = input_feature.shape[channel_axis]

	se_feature = GlobalAveragePooling1D()(input_feature)
	se_feature = Reshape((1, channel))(se_feature)
	assert se_feature.shape[1:] == (1,channel)
	# se_feature = Dense(channel // ratio,
	# 					   activation='relu',
	# 					   kernel_initializer='he_normal',
	# 					   use_bias=True,
	# 					   bias_initializer='zeros')(se_feature)
	se_feature=Conv1D(channel // ratio,1,padding='same', data_format='channels_last', use_bias=False)(se_feature)
	se_feature=Activation('relu')(se_feature)
	assert se_feature.shape[1:] == (1,channel // ratio)
	# se_feature = Dense(channel,
	# 				   activation='sigmoid',
	# 				   kernel_initializer='he_normal',
	# 				   use_bias=True,
	# 				   bias_initializer='zeros')(se_feature)
	se_feature = Conv1D(channel, 1, padding='same', data_format='channels_last', use_bias=False)(se_feature)
	se_feature = Activation('sigmoid')(se_feature)
	assert se_feature.shape[1:] == (1,channel)
	se_feature = multiply([input_feature, se_feature])
	return se_feature


def se_block1(input_feature, ratio=8):
	"""Squeeze-and-Excitation(SE) block.

	As described in https://arxiv.org/abs/1709.01507
	The implementation is taken from https://github.com/kobiso/CBAM-keras
	"""
	channel_axis = 1 if K.image_data_format() == "channels_first" else -1
	channel = input_feature.shape[channel_axis]

	se_feature = GlobalAveragePooling2D()(input_feature)
	se_feature = Reshape((1, 1, channel))(se_feature)
	assert se_feature.shape[1:] == (1,1,channel)
	se_feature = Dense(channel // ratio,
					   activation='relu',
					   kernel_initializer='he_normal',
					   use_bias=True,
					   bias_initializer='zeros')(se_feature)
	assert se_feature.shape[1:] == (1,1,channel//ratio)
	se_feature = Dense(channel,
					   activation='sigmoid',
					   kernel_initializer='he_normal',
					   use_bias=True,
					   bias_initializer='zeros')(se_feature)
	assert se_feature.shape[1:] == (1,1,channel)
	if K.image_data_format() == 'channels_first':
		se_feature = Permute((3, 1, 2))(se_feature)

	return se_feature

def cse_block1(input_feature, ratio=8):
	"""Squeeze-and-Excitation(SE) block.

	As described in https://arxiv.org/abs/1709.01507
	The implementation is taken from https://github.com/kobiso/CBAM-keras
	"""
	# channel_axis = 1 if K.image_data_format() == "channels_first" else -1
	# channel = input_feature.shape[channel_axis]
	channel_axis = 1 if K.image_data_format() == "channels_first" else -1
	channel = input_feature.shape[channel_axis]

	se_feature = GlobalAveragePooling1D()(input_feature)
	se_feature = Reshape((1, channel))(se_feature)
	assert se_feature.shape[1:] == (1,channel)
	se_feature = Dense(channel // ratio,
						   activation='relu',
						   kernel_initializer='he_normal',
						   use_bias=True,
						   bias_initializer='zeros')(se_feature)
	assert se_feature.shape[1:] == (1,channel // ratio)
	se_feature = Dense(channel,
					   activation='sigmoid',
					   kernel_initializer='he_normal',
					   use_bias=True,
					   bias_initializer='zeros')(se_feature)
	assert se_feature.shape[1:] == (1,channel)
	# se_feature = multiply([input_feature, se_feature])
	return se_feature


def PConv2d(input_feature,channel,size):
	block1,block2=tf.split(input_feature,[channel//4,channel-channel//4],-1)
	block1=Conv2D(channel//4,size,data_format='channels_last',use_bias=False, padding='same')(block1)
	block3=Concatenate(axis=-1)([block1,block2])
	return block3
def PConv1d(input_feature,channel,size):
	block1,block2=tf.split(input_feature,[channel//4,channel-channel//4],-1)
	block1=Conv1D(channel//4,size,data_format='channels_last',use_bias=False, padding='same')(block1)
	block3=Concatenate(axis=-1)([block1,block2])
	return block3
def mse_block(x1,x2,x3, ratio=8):
	# channel_axis = 1 if K.image_data_format() == "channels_first" else -1
	# channel = input_feature.shape[channel_axis]
	# channel=16
	# x1 = Conv2D(16, (16, 1), use_bias=False, padding='same')(input_feature)
	# x1 = BatchNormalization()(x1)
	# x1 = Activation('elu')(x1)
	#
	# x2 = Conv2D(16, (32, 1), use_bias=False, padding='same')(input_feature)
	# x2 = BatchNormalization()(x2)
	# x2 = Activation('elu')(x2)
	#
	# x3 = Conv2D(16, (64, 1), use_bias=False, padding='same')(input_feature)
	# x3 = BatchNormalization()(x3)
	# x3 = Activation('elu')(x3)
	# x = Concatenate(axis=-1)([x1, x2, x3])
	channel=16
	x=Concatenate(axis=-1)([x1,x2,x3])
	# y=tf.reduce_sum(x,axis=-1,keepdims=True)
	se_feature = GlobalAveragePooling2D()(x)
	se_feature = Reshape((1, 1, channel*3))(se_feature)
	se_feature = Dense(channel*3 // ratio,
					   activation='relu',
					   kernel_initializer='he_normal',
					   use_bias=True,
					   bias_initializer='zeros')(se_feature)
	se_feature = Dense(channel*3,
					   activation='sigmoid',
					   kernel_initializer='he_normal',
					   use_bias=True,
					   bias_initializer='zeros')(se_feature)
	se_feature=Reshape([1,1,channel,3])(se_feature)
	se_feature=softmax(se_feature)
	# x=Reshape([1125,1,channel,3])(x)
	# se_feature=multiply([x, se_feature])
	# se_feature=tf.reduce_sum(se_feature,axis=-1)
	se_feature1=se_feature[:,:,:,:, 0]
	se_feature2 = se_feature[:,:, :, :, 1]
	se_feature3 = se_feature[:,:, :, :, 2]
	se_feature1 = multiply([x1, se_feature1])

	se_feature2 = multiply([x2, se_feature2])

	se_feature3 = multiply([x3, se_feature3])
	#
	# se_feature=se_feature1+se_feature2+se_feature3
	return se_feature1,se_feature2,se_feature3
def mse_block1d(x1,x2,x3, ratio=8):
	# channel_axis = 1 if K.image_data_format() == "channels_first" else -1
	# channel = input_feature.shape[channel_axis]
	# channel=16
	# x1 = Conv2D(16, (16, 1), use_bias=False, padding='same')(input_feature)
	# x1 = BatchNormalization()(x1)
	# x1 = Activation('elu')(x1)
	#
	# x2 = Conv2D(16, (32, 1), use_bias=False, padding='same')(input_feature)
	# x2 = BatchNormalization()(x2)
	# x2 = Activation('elu')(x2)
	#
	# x3 = Conv2D(16, (64, 1), use_bias=False, padding='same')(input_feature)
	# x3 = BatchNormalization()(x3)
	# x3 = Activation('elu')(x3)
	# x = Concatenate(axis=-1)([x1, x2, x3])
	channel=32
	x=Concatenate(axis=-1)([x1,x2,x3])
	# y=tf.reduce_sum(x,axis=-1,keepdims=True)
	se_feature = GlobalAveragePooling1D()(y)
	se_feature = Reshape((1, channel))(se_feature)

	se_feature = Dense(channel*3 // ratio,
					   activation='relu',
					   kernel_initializer='he_normal',
					   use_bias=True,
					   bias_initializer='zeros')(se_feature)
	# se_feature=Conv2D(channel // ratio,(1,1))(se_feature)

	se_feature = Dense(channel*3,
					   activation='sigmoid',
					   kernel_initializer='he_normal',
					   use_bias=True,
					   bias_initializer='zeros')(se_feature)
	se_feature=Reshape([1,channel,3])(se_feature)
	se_feature=softmax(se_feature)
	x=Reshape([1125,channel,3])(x)
	se_feature=multiply([x, se_feature])
	se_feature=tf.reduce_sum(se_feature,axis=-1)
	se_feature1=se_feature[:,0:1,:,: ]
	se_feature2 = se_feature[:, 1:2, :, :]
	se_feature3 = se_feature[:, 2:3, :, :]

	se_feature1 = multiply([x1, se_feature1])

	se_feature2 = multiply([x2, se_feature2])

	se_feature3 = multiply([x3, se_feature3])
	#
	# se_feature=se_feature1+se_feature2+se_feature3
	return se_feature1,se_feature2,se_feature3

#%% Convolutional block attention module
def cbam_block(cbam_feature, ratio=8):
	""" Convolutional Block Attention Module(CBAM) block.

	As described in https://arxiv.org/abs/1807.06521
	The implementation is taken from https://github.com/kobiso/CBAM-keras
	"""

	block2a1 = channel_attention1d(cbam_feature, ratio)
	block2b1 = channel_attention1d(cbam_feature, ratio)
	block2c1 = channel_attention1d(cbam_feature, ratio)
	block2a1 = Permute((2, 1))(block2a1)
	a1 = tf.matmul(block2b1, block2a1)
	softmax = Activation('softmax')(a1)
	block3 = tf.matmul(softmax, block2c1)
	block3 = tf.add(block3, cbam_feature)

	# cbam_feature = spatial_attention(cbam_feature)
	return block3


def channel_attention1d(input_feature, ratio=8):
	channel_axis = 1 if K.image_data_format() == "channels_first" else -1
	# 	channel = input_feature._keras_shape[channel_axis]
	channel = input_feature.shape[channel_axis]

	shared_layer_one = Dense(channel // ratio,
							 activation='elu',
							 kernel_initializer='he_normal',
							 use_bias=True,
							 bias_initializer='zeros')
	shared_layer_two = Dense(channel,
							 kernel_initializer='he_normal',
							 use_bias=True,
							 bias_initializer='zeros')

	avg_pool = GlobalAveragePooling1D()(input_feature)
	avg_pool = Reshape((1, channel))(avg_pool)
	assert avg_pool.shape[1:] == (1,  channel)
	avg_pool = shared_layer_one(avg_pool)
	assert avg_pool.shape[1:] == (1,  channel // ratio)
	avg_pool = shared_layer_two(avg_pool)
	assert avg_pool.shape[1:] == (1, channel)

	max_pool = GlobalMaxPooling1D()(input_feature)
	max_pool = Reshape((1,  channel))(max_pool)
	assert max_pool.shape[1:] == (1,  channel)
	max_pool = shared_layer_one(max_pool)
	assert max_pool.shape[1:] == (1,  channel // ratio)
	max_pool = shared_layer_two(max_pool)
	assert max_pool.shape[1:] == (1,  channel)

	cbam_feature = Add()([avg_pool, max_pool])
	cbam_feature = Activation('sigmoid')(cbam_feature)

	if K.image_data_format() == "channels_first":
		cbam_feature = Permute((3, 1, 2))(cbam_feature)

	return multiply([input_feature, cbam_feature])


def channel_attention(input_feature, ratio=8):
	channel_axis = 1 if K.image_data_format() == "channels_first" else -1
# 	channel = input_feature._keras_shape[channel_axis]
	channel = input_feature.shape[channel_axis]

	shared_layer_one = Dense(channel//ratio,
							 activation='relu',
							 kernel_initializer='he_normal',
							 use_bias=True,
							 bias_initializer='zeros')
	shared_layer_two = Dense(channel,
							 kernel_initializer='he_normal',
							 use_bias=True,
							 bias_initializer='zeros')
	
	avg_pool = GlobalAveragePooling2D()(input_feature)    
	avg_pool = Reshape((1,1,channel))(avg_pool)
	assert avg_pool.shape[1:] == (1,1,channel)
	avg_pool = shared_layer_one(avg_pool)
	assert avg_pool.shape[1:] == (1,1,channel//ratio)
	avg_pool = shared_layer_two(avg_pool)
	assert avg_pool.shape[1:] == (1,1,channel)
	
	max_pool = GlobalMaxPooling2D()(input_feature)
	max_pool = Reshape((1,1,channel))(max_pool)
	assert max_pool.shape[1:] == (1,1,channel)
	max_pool = shared_layer_one(max_pool)
	assert max_pool.shape[1:] == (1,1,channel//ratio)
	max_pool = shared_layer_two(max_pool)
	assert max_pool.shape[1:] == (1,1,channel)
	
	cbam_feature = Add()([avg_pool,max_pool])
	cbam_feature = Activation('sigmoid')(cbam_feature)
	
	if K.image_data_format() == "channels_first":
		cbam_feature = Permute((3, 1, 2))(cbam_feature)
	
	return multiply([input_feature, cbam_feature])

def spatial_attention(input_feature):
	kernel_size = 7
	
	if K.image_data_format() == "channels_first":
		channel = input_feature.shape[1]
		cbam_feature = Permute((2,3,1))(input_feature)
	else:
		channel = input_feature.shape[-1]
		cbam_feature = input_feature
	
	avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
	assert avg_pool.shape[-1] == 1
	max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
	assert max_pool.shape[-1] == 1
	concat = Concatenate(axis=3)([avg_pool, max_pool])
	assert concat.shape[-1] == 2
	cbam_feature = Conv2D(filters = 1,
					kernel_size=(1,7),
					strides=1,
					padding='same',
					activation='sigmoid',
					kernel_initializer='he_normal',
					use_bias=False)(concat)	
	assert cbam_feature.shape[-1] == 1
	
	if K.image_data_format() == "channels_first":
		cbam_feature = Permute((3, 1, 2))(cbam_feature)
		
	return multiply([input_feature, cbam_feature])


def channel_add(input_feature,input_feature1,input_feature2, ratio=8):
	channel_axis = 1 if K.image_data_format() == "channels_first" else -1
	# 	channel = input_feature._keras_shape[channel_axis]
	channel = input_feature.shape[channel_axis]

	avg_pool = GlobalAveragePooling2D()(input_feature)
	avg_pool = Reshape((1, 1, channel))(avg_pool)
	assert avg_pool.shape[1:] == (1, 1, channel)
	avg_pool = Dense(channel // ratio,
							 activation='elu',
							 kernel_initializer='he_normal',
							 use_bias=True,
							 bias_initializer='zeros')(avg_pool)
	assert avg_pool.shape[1:] == (1, 1, channel // ratio)
	avg_pool = Dense(channel,
							 activation='sigmoid',
							 kernel_initializer='he_normal',
							 use_bias=True,
							 bias_initializer='zeros')(avg_pool)
	assert avg_pool.shape[1:] == (1, 1, channel)

	avg_pool1 = GlobalAveragePooling2D()(input_feature1)
	avg_pool1 = Reshape((1, 1, channel))(avg_pool1)

	avg_pool1 = Dense(channel // ratio,
							 activation='elu',
							 kernel_initializer='he_normal',
							 use_bias=True,
							 bias_initializer='zeros')(avg_pool1)

	avg_pool1 = Dense(channel,
							 activation='sigmoid',
							 kernel_initializer='he_normal',
							 use_bias=True,
							 bias_initializer='zeros')(avg_pool1)

	avg_pool2 = GlobalAveragePooling2D()(input_feature2)
	avg_pool2 = Reshape((1, 1, channel))(avg_pool2)

	avg_pool2 = Dense(channel // ratio,
							 activation='elu',
							 kernel_initializer='he_normal',
							 use_bias=True,
							 bias_initializer='zeros')(avg_pool2)

	avg_pool2 = Dense(channel,
							 activation='sigmoid',
							 kernel_initializer='he_normal',
							 use_bias=True,
							 bias_initializer='zeros')(avg_pool2)
	output = avg_pool+avg_pool1+avg_pool2
	avg_pool = multiply([input_feature, output])
	avg_pool1 = multiply([input_feature1, output])
	avg_pool2 = multiply([input_feature2, output])
	return avg_pool,avg_pool1,avg_pool2



def eca_block(input_feature,input_feature1,input_feature2,input_feature3,b=1, gamma=2):
	channel_axis = 1 if K.image_data_format() == "channels_first" else -1
	channel = input_feature.shape[channel_axis]
	kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
	kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

	avg_pool = GlobalAveragePooling2D()(input_feature)
	x = Reshape((-1, 1))(avg_pool)
	x = Conv1D(1, kernel_size=1, padding="same", use_bias=False)(x)
	x = Activation('relu')(x)
	x = Reshape((1, 1, -1))(x)

	avg_pool = GlobalAveragePooling2D()(input_feature1)
	x1 = Reshape((-1, 1))(avg_pool)
	x1 = Conv1D(1, kernel_size=1, padding="same", use_bias=False)(x1)
	x1 = Activation('relu')(x1)
	x1 = Reshape((1, 1, -1))(x1)

	avg_pool = GlobalAveragePooling2D()(input_feature2)
	x2 = Reshape((-1, 1))(avg_pool)
	x2 = Conv1D(1, kernel_size=1, padding="same", use_bias=False)(x2)
	x2 = Activation('relu')(x2)
	x2 = Reshape((1, 1, -1))(x2)

	avg_pool = GlobalAveragePooling2D()(input_feature3)
	x3 = Reshape((-1, 1))(avg_pool)
	x3 = Conv1D(1, kernel_size=1, padding="same", use_bias=False)(x3)
	x3 = Activation('relu')(x3)
	x3 = Reshape((1, 1, -1))(x3)

	se_feature = Concatenate(axis=-1)([x, x1, x2,x3])
	se_feature = Reshape([1, 1, channel, 4])(se_feature)
	se_feature = softmax(se_feature)
	x = se_feature[:, :, :, :, 0]
	x1 = se_feature[:, :, :, :, 1]
	x2 = se_feature[:, :, :, :, 2]
	x3 = se_feature[:, :, :, :, 3]

	output = multiply([input_feature, x])
	output1 = multiply([input_feature1, x1])
	output2 = multiply([input_feature2, x2])
	output3 = multiply([input_feature3, x3])

	return output,output1,output2,output3




