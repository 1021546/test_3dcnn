import scipy.io.wavfile as wav
import numpy as np
import os
import sys
lib_path = os.path.abspath(os.path.join('..'))
print(lib_path)
sys.path.append(lib_path)
import speechpy

import keras
from keras.layers import (Activation, Conv3D, Dense, Dropout, Flatten,
                          MaxPooling3D, Reshape)
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt

all_mfec_shape=np.empty(shape=[0])
all_mfec=np.empty(shape=[0,40,30])

for i in range(1,7):
	speaker_mfec=np.empty(shape=[0,40])
	for j in range(0,6):
		for k in range(1,6):
			file_path = './wav/'+str(i)+'/'+str(j)+'_'+str(k)+'.wav'
			file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),file_path)
			fs, signal = wav.read(file_name)
			print(type(signal))
			print(signal.shape)
			#signal = signal[:,0]

			# Example of pre-emphasizing.
			signal_preemphasized = speechpy.processing.preemphasis(signal, cof=0.98)

			# Example of staching frames
			frames = speechpy.processing.stack_frames(signal, sampling_frequency=fs, frame_length=0.020, frame_stride=0.01, filter=lambda x: np.ones((x,)),
			         zero_padding=True)

			# Example of extracting power spectrum
			power_spectrum = speechpy.processing.power_spectrum(frames, fft_points=512)
			print('power spectrum shape=', power_spectrum.shape)

			############# Extract MFCC features #############
			mfcc = speechpy.feature.mfcc(signal, sampling_frequency=fs, frame_length=0.020, frame_stride=0.01,
			             num_filters=40, fft_length=512, low_frequency=0, high_frequency=None)
			mfcc_cmvn = speechpy.processing.cmvnw(mfcc,win_size=301,variance_normalization=True)
			print('mfcc(mean + variance normalized) feature shape=', mfcc_cmvn.shape)

			mfcc_feature_cube = speechpy.feature.extract_derivative_feature(mfcc)
			print('mfcc feature cube shape=', mfcc_feature_cube.shape)

			############# Extract logenergy features #############
			logenergy = speechpy.feature.lmfe(signal, sampling_frequency=fs, frame_length=0.020, frame_stride=0.01,
			             num_filters=40, fft_length=512, low_frequency=0, high_frequency=None)
			logenergy_feature_cube = speechpy.feature.extract_derivative_feature(logenergy)
			print('logenergy features=', logenergy.shape)
			# print('logenergy features=', logenergy.shape[0])
			# all_mfec_shape=np.hstack((all_mfec_shape,logenergy.shape[0]))
			# print(logenergy[0:99])
			# print(logenergy[0:99].shape)
			# print(type(logenergy[0:99]))
			# input()
			speaker_mfec=np.vstack((speaker_mfec,logenergy[0:99]))
	speaker_mfec = np.reshape(speaker_mfec, (99, 40, 30))
	# print(speaker_mfec)
	# print(speaker_mfec.shape)
	# print(type(speaker_mfec))
	all_mfec=np.vstack((all_mfec,speaker_mfec))
	# print(all_mfec)
	# print(all_mfec.shape)
	# print(type(all_mfec))

# print(all_mfec_shape)
# print(all_mfec_shape.shape)
# print(np.amin(all_mfec_shape))

# print(speaker_mfec)
# print(speaker_mfec.shape)
# print(type(speaker_mfec))

all_mfec = np.reshape(all_mfec, (6, 99, 40, 30))

# print(all_mfec[0][:][:])
# print(all_mfec[0][:][:].shape)
# print(type(all_mfec[0][:][:]))
# input()

# print(all_mfec[0])
# print(all_mfec[0].shape)
# print(type(all_mfec[0]))
# input()

x_train=all_mfec

y_train=np.zeros(6)

for i in range(0,6):
	y_train[i]=i

# print(y_train)
# print(y_train.shape)
# print(type(y_train))

# one speaker , one file duplicate
if False:
	one_speaker=all_mfec[0][:][:]
	print("one_speaker")
	print(one_speaker)
	print(one_speaker.shape)
	print(type(one_speaker))
	input()

	one_file=one_speaker[..., 1]
	print("one_file")
	print(one_file)
	print(one_file.shape)
	print(type(one_file))
	input()

	one_file = np.reshape(one_file, (99, 40, 1))

	x_test=np.empty(shape=[0,40, 30])
	x_test = np.repeat(one_file,30,axis=2)
	

	print(x_test)
	print(x_test.shape)
	print(type(x_test))
	input()

	x_test = np.reshape(x_test, (1, 99, 40, 30))

	# print(x_test)
	# print(x_test.shape)
	# print(type(x_test))
	# input()

	y_test=np.zeros(1)

# one speaker , every file
if True:
	x_test=np.empty(shape=[0,40,30])
	
	x_test=all_mfec[0]
	# print("x_test")
	# print(x_test)
	# print(x_test.shape)
	# print(type(x_test))

	x_test = np.reshape(x_test, (1, 99, 40, 30))
	# print(x_test)
	# print(x_test.shape)
	# print(type(x_test))
	# input()

	y_test=np.zeros(1)

# every speaker , part of file
if False:
	all_mfec=np.empty(shape=[0,40,30])
	speaker_mfec=np.empty(shape=[0,40])
	
	for i in range(0,6):		
		for j in range(1,6):
			file_path = './wav/1/'+str(i)+'_'+str(j)+'.wav'
			file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),file_path)
			fs, signal = wav.read(file_name)
			print(type(signal))
			print(signal.shape)
			#signal = signal[:,0]

			# Example of pre-emphasizing.
			signal_preemphasized = speechpy.processing.preemphasis(signal, cof=0.98)

			# Example of staching frames
			frames = speechpy.processing.stack_frames(signal, sampling_frequency=fs, frame_length=0.020, frame_stride=0.01, filter=lambda x: np.ones((x,)),
			         zero_padding=True)

			# Example of extracting power spectrum
			power_spectrum = speechpy.processing.power_spectrum(frames, fft_points=512)
			print('power spectrum shape=', power_spectrum.shape)

			############# Extract MFCC features #############
			mfcc = speechpy.feature.mfcc(signal, sampling_frequency=fs, frame_length=0.020, frame_stride=0.01,
			             num_filters=40, fft_length=512, low_frequency=0, high_frequency=None)
			mfcc_cmvn = speechpy.processing.cmvnw(mfcc,win_size=301,variance_normalization=True)
			print('mfcc(mean + variance normalized) feature shape=', mfcc_cmvn.shape)

			mfcc_feature_cube = speechpy.feature.extract_derivative_feature(mfcc)
			print('mfcc feature cube shape=', mfcc_feature_cube.shape)

			############# Extract logenergy features #############
			logenergy = speechpy.feature.lmfe(signal, sampling_frequency=fs, frame_length=0.020, frame_stride=0.01,
			             num_filters=40, fft_length=512, low_frequency=0, high_frequency=None)
			logenergy_feature_cube = speechpy.feature.extract_derivative_feature(logenergy)
			print('logenergy features=', logenergy.shape)
			# print('logenergy features=', logenergy.shape[0])
			# all_mfec_shape=np.hstack((all_mfec_shape,logenergy.shape[0]))
			# print(logenergy[0:99])
			# print(logenergy[0:99].shape)
			# print(type(logenergy[0:99]))
			# input()
			speaker_mfec=np.vstack((speaker_mfec,logenergy[0:99]))

	# print(speaker_mfec)
	# print(speaker_mfec.shape)
	# print(type(speaker_mfec))
	# input()

	speaker_mfec = np.reshape(speaker_mfec, (99, 40, 30))

	# print(speaker_mfec)
	# print(speaker_mfec.shape)
	# print(type(speaker_mfec))

	all_mfec = np.reshape(speaker_mfec, (1, 99, 40, 30))

	x_test=all_mfec

	y_test=np.zeros(1)


# convert class vectors to binary class matrices - this is for use in the
# categorical_crossentropy loss below
num_classes=6
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = np.transpose(x_train[None, :, :, :, :], axes=(1, 4, 2, 3, 0))
x_test = np.transpose(x_test[None, :, :, :, :], axes=(1, 4, 2, 3, 0))

print("x_train")
# print(x_train)
print(x_train.shape)
# print(type(x_train))
# input()

print("x_test")
# print(x_test)
print(x_test.shape)
# print(type(x_test))
# input()


# Define model
model = Sequential()
# model.add(Reshape())
model.add(Conv3D(32, kernel_size=(3, 3, 3), input_shape=(30,99,40,1), padding='same'))
model.add(Conv3D(32, kernel_size=(3, 3, 3), padding='same'))
model.add(PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None))
model.add(Conv3D(32, kernel_size=(3, 3, 3), padding='same'))
model.add(PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None))
model.add(MaxPooling3D(pool_size=(3, 3, 3), padding='same'))
model.add(Dropout(0.25))

model.add(Conv3D(64, kernel_size=(3, 3, 3), padding='same'))
model.add(PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None))
model.add(Conv3D(64, kernel_size=(3, 3, 3), padding='same'))
model.add(PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None))
model.add(MaxPooling3D(pool_size=(3, 3, 3), padding='same'))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=categorical_crossentropy,
          optimizer=Adam(), metrics=['accuracy'])
model.summary()
# plot_model(model, show_shapes=True, to_file='model_prelu_1.png')

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=20, epochs=10, verbose=1, shuffle=True)

loss, acc = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', loss)
print('Test accuracy:', acc)

def plot_history(history, result_dir):
	plt.plot(history.history['acc'], marker='.')
	plt.plot(history.history['val_acc'], marker='.')
	plt.title('model accuracy')
	plt.xlabel('epoch')
	plt.ylabel('accuracy')
	plt.grid()
	plt.legend(['acc', 'val_acc'], loc='lower right')
	plt.savefig(os.path.join(result_dir, 'model_accuracy_prelu_1.png'))
	plt.close()

	plt.plot(history.history['loss'], marker='.')
	plt.plot(history.history['val_loss'], marker='.')
	plt.title('model loss')
	plt.xlabel('epoch')
	plt.ylabel('loss')
	plt.grid()
	plt.legend(['loss', 'val_loss'], loc='upper right')
	plt.savefig(os.path.join(result_dir, 'model_loss_prelu_1.png'))
	plt.close()

# plot_history(history,os.path.dirname(os.path.abspath(__file__)))

# model.save('my_model_prelu_1.h5')
# del model