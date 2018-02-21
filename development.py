import scipy.io.wavfile as wav
import numpy as np
import os
import sys
lib_path = os.path.abspath(os.path.join('..'))
print(lib_path)
sys.path.append(lib_path)
import speechpy
import keras
from keras.models import Sequential
from keras.layers.advanced_activations import PReLU
from keras.layers import (Activation, Conv3D, Dense, Dropout, Flatten,
                          MaxPooling3D, Reshape)
from keras.optimizers import SGD, Adam
from keras.losses import categorical_crossentropy
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping


# training data
temp_frame=1000
temp_i=-1
temp_j=-1
temp_k=-1

all_speaker_mfec=np.empty(shape=[0,20,110,40])
one_speaker_mfec=np.empty(shape=[0,110,40])

for i in range(1,6):
	for j in range(1,5):
		one_train_mfec=np.empty(shape=[0,40])
		for k in range(0,20):
			file_path = './development_wav/'+str(i)+'/'+'train'+str(j)+'/'+str(k)+'.wav'
			file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_path)
			fs, signal = wav.read(file_name)

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

			one_train_mfec=np.vstack((one_train_mfec,logenergy[0:110]))

			# decide minimum frame
			# if(temp_frame>(logenergy.shape[0])):
			# 	temp_frame=logenergy.shape[0]
			# 	temp_i=i
			# 	temp_j=j
			# 	temp_k=k
		one_train_mfec = np.reshape(one_train_mfec, (20, 110, 40))
		one_speaker_mfec=np.vstack((one_speaker_mfec,one_train_mfec))

# print(temp_frame)
# print(temp_i)
# print(temp_j)
# print(temp_k)

# print(one_speaker_mfec)
# print(one_speaker_mfec.shape)
# print(type(one_speaker_mfec))
# input()
all_speaker_mfec = np.reshape(one_speaker_mfec, (20, 20, 110, 40))
# print(all_speaker_mfec)
# print(all_speaker_mfec.shape)
# print(type(all_speaker_mfec))
# input()

x_train=all_speaker_mfec

# np.save('./temp/x_train.npy', x_train)
# x_train = np.load('./temp/x_train.npy')

y_train=np.zeros(20)
for i in range(0,20):
	label=i//4
	y_train[i]=label

# np.save('./temp/y_train.npy', y_train)
# y_train = np.load('./temp/y_train.npy')

# print(y_train)
# print(y_train.shape)
# print(type(y_train))
# input()


# testing data
temp_frame=1000
temp_i=-1
temp_j=-1

all_speaker_mfec=np.empty(shape=[0,20,110,40])
one_speaker_mfec=np.empty(shape=[0,110,40])

for i in range(1,6):
	one_test_mfec=np.empty(shape=[0,40])
	for j in range(0,20):
		file_path = './development_wav/'+str(i)+'/test/'+str(j)+'.wav'
		file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_path)
		fs, signal = wav.read(file_name)

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

		one_test_mfec=np.vstack((one_test_mfec,logenergy[0:110]))

		# decide minimum frame
		# if(temp_frame>(logenergy.shape[0])):
		# 	temp_frame=logenergy.shape[0]
		# 	temp_i=i
		# 	temp_j=j
	one_test_mfec = np.reshape(one_test_mfec, (20, 110, 40))
	one_speaker_mfec=np.vstack((one_speaker_mfec,one_test_mfec))

# print(temp_frame)
# print(temp_i)
# print(temp_j)

# print(one_speaker_mfec)
# print(one_speaker_mfec.shape)
# print(type(one_speaker_mfec))
# input()
all_speaker_mfec = np.reshape(one_speaker_mfec, (5, 20, 110, 40))
# print(all_speaker_mfec)
# print(all_speaker_mfec.shape)
# print(type(all_speaker_mfec))
# input()

x_test=all_speaker_mfec

# np.save('./temp/x_test.npy', x_test)
# x_test = np.load('./temp/x_test.npy')

y_test=np.zeros(5)
for i in range(0,5):
	y_test[i]=i


# np.save('./temp/y_test.npy', y_test)
# y_test = np.load('./temp/y_test.npy')

# print(y_test)
# print(y_test.shape)
# print(type(y_test))
# input()



num_classes = 5
batch_size = 128
epochs = 30
# convert class vectors to binary class matrices - this is for use in the
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

early_stopping = EarlyStopping(monitor = 'val_loss', patience = 3)
sgd = SGD(lr=0.01,momentum=0.0,decay=0.0,nesterov=False)

# Define model
model = Sequential()
model.add(Reshape((20,110,40,1) , input_shape=(20,110,40)))
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

# model = Sequential()
# model.add(Reshape((20,110,40,1) , input_shape=(20,110,40)))
# model.add(Conv3D(16, kernel_size=(3, 1, 5), strides=(1, 1, 1), padding='same'))
# model.add(PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None))
# model.add(Conv3D(16, kernel_size=(3, 9, 1), strides=(1, 2, 1), padding='same'))
# model.add(PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None))
# model.add(MaxPooling3D(pool_size=(1, 1, 2), strides=(1, 1, 2), padding='valid'))
# # model.add(Dropout(0.25))

# model.add(Conv3D(32, kernel_size=(3, 1, 4), strides=(1, 1, 1), padding='same'))
# model.add(PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None))
# model.add(Conv3D(32, kernel_size=(3, 8, 1), strides=(1, 2, 1), padding='same'))
# model.add(PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None))
# model.add(MaxPooling3D(pool_size=(1, 1, 2), strides=(1, 2, 1), padding='valid'))


# model.add(Conv3D(64, kernel_size=(3, 1, 3), strides=(1, 1, 1), padding='same'))
# model.add(PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None))
# model.add(Conv3D(64, kernel_size=(3, 7, 1), strides=(1, 1, 1), padding='same'))
# model.add(PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None))
# # model.add(MaxPooling3D(pool_size=(3, 3, 3), strides=(1, 1, 1), padding='same'))
# # model.add(Dropout(0.25))

# model.add(Conv3D(128, kernel_size=(3, 1, 3), strides=(1, 1, 1), padding='same'))
# model.add(PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None))
# model.add(Conv3D(128, kernel_size=(3, 7, 1), strides=(1, 1, 1), padding='same'))
# model.add(PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None))


# model.add(Conv3D(128, kernel_size=(4, 3, 3), strides=(1, 1, 1), padding='same'))
# model.add(PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None))

# model.add(Flatten())
# model.add(Dense(512))
# model.add(PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=categorical_crossentropy,
          optimizer=sgd, metrics=['accuracy'])
model.summary()
# plot_model(model, show_shapes=True, to_file='development_model.png')

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=batch_size, 
	epochs=epochs, verbose=1, shuffle=True, callbacks=[early_stopping])


loss, acc = model.evaluate(x_test, y_test, verbose=1)

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
	plt.show()
	plt.savefig(os.path.join(result_dir, 'development_accuracy.png'))
	plt.close()

	plt.plot(history.history['loss'], marker='.')
	plt.plot(history.history['val_loss'], marker='.')
	plt.title('model loss')
	plt.xlabel('epoch')
	plt.ylabel('loss')
	plt.grid()
	plt.legend(['loss', 'val_loss'], loc='upper right')
	plt.show()
	plt.savefig(os.path.join(result_dir, 'development_loss.png'))
	plt.close()

# plot_history(history,os.path.dirname(os.path.abspath(__file__)))

# model.save('my_development_model.h5')
# del model