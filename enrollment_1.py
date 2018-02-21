from keras.models import load_model
import scipy.io.wavfile as wav
import numpy as np
import os
import sys
lib_path = os.path.abspath(os.path.join('..'))
print(lib_path)
sys.path.append(lib_path)
import speechpy
import keras
from keras import backend as K
# with a Sequential model
import matplotlib.pyplot as plt

background_model = load_model('./development_data/my_development_model_1.h5')
# background_model.summary()

# enrollment data
# temp_frame=1000
# temp_i=-1
# temp_j=-1
# temp_k=-1

all_speaker_mfec=np.empty(shape=[0,20,99,40])
one_speaker_mfec=np.empty(shape=[0,99,40])

for i in range(1,5):
	for j in range(1,16):
		one_enrollment_mfec=np.empty(shape=[0,40])
		for k in range(0,20):
			file_path = './enrollment_wav/'+str(i)+'/'+str(j)+'/'+str(j)+'.wav'
			file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_path)
			fs, signal = wav.read(file_name)
			# print(i)
			# print(j)
			# print(k)
			# input()

			# Example of pre-emphasizing.
			signal_preemphasized = speechpy.processing.preemphasis(signal, cof=0.98)

			# Example of staching frames
			frames = speechpy.processing.stack_frames(signal, sampling_frequency=fs, frame_length=0.020, frame_stride=0.01, filter=lambda x: np.ones((x,)),
			         zero_padding=True)

			# Example of extracting power spectrum
			power_spectrum = speechpy.processing.power_spectrum(frames, fft_points=512)
			# print('power spectrum shape=', power_spectrum.shape)

			############# Extract MFCC features #############
			mfcc = speechpy.feature.mfcc(signal, sampling_frequency=fs, frame_length=0.020, frame_stride=0.01,
			             num_filters=40, fft_length=512, low_frequency=0, high_frequency=None)
			mfcc_cmvn = speechpy.processing.cmvnw(mfcc,win_size=301,variance_normalization=True)
			# print('mfcc(mean + variance normalized) feature shape=', mfcc_cmvn.shape)

			mfcc_feature_cube = speechpy.feature.extract_derivative_feature(mfcc)
			# print('mfcc feature cube shape=', mfcc_feature_cube.shape)

			############# Extract logenergy features #############
			logenergy = speechpy.feature.lmfe(signal, sampling_frequency=fs, frame_length=0.020, frame_stride=0.01,
			             num_filters=40, fft_length=512, low_frequency=0, high_frequency=None)
			logenergy_feature_cube = speechpy.feature.extract_derivative_feature(logenergy)
			# print('logenergy features=', logenergy.shape)

			one_enrollment_mfec=np.vstack((one_enrollment_mfec,logenergy[0:99]))

			# decide minimum frame
			# if(temp_frame>(logenergy.shape[0])):
			# 	temp_frame=logenergy.shape[0]
			# 	temp_i=i
			# 	temp_j=j
			# 	temp_k=k
		one_enrollment_mfec = np.reshape(one_enrollment_mfec, (20, 99, 40))
		one_speaker_mfec=np.vstack((one_speaker_mfec,one_enrollment_mfec))

# print(temp_frame)
# print(temp_i)
# print(temp_j)
# print(temp_k)


# print(one_speaker_mfec)
# print(one_speaker_mfec.shape)
# print(type(one_speaker_mfec))
# input()
all_speaker_mfec = np.reshape(one_speaker_mfec, (60, 20, 99, 40))
# print(all_speaker_mfec)
# print(all_speaker_mfec.shape)
# print(type(all_speaker_mfec))
# input()

x_train=all_speaker_mfec
np.save('./enrollment_data/x_train.npy', x_train)
# x_train = np.load('./enrollment_data/x_train.npy')

y_train=np.zeros(60, dtype=np.int)
for i in range(0,60):
	label=i//15
	y_train[i]=label

np.save('./enrollment_data/y_train.npy', y_train)
# y_train = np.load('./enrollment_data/y_train.npy')

# print(y_train)
# print(y_train.shape)
# print(type(y_train))
# input()


num_classes = 5
batch_size = 128
epochs = 20

# convert class vectors to binary class matrices - this is for use in the
y_train = keras.utils.to_categorical(y_train, num_classes)

# history = background_model.fit(x_train, y_train, validation_data=None, batch_size=batch_size, 
# 	epochs=epochs, verbose=1, shuffle=True)


get_26th_layer_output = K.function([background_model.layers[0].input, K.learning_phase()],
                                  [background_model.layers[26].output])


layer_output = get_26th_layer_output([x_train, 1])[0]
# print(layer_output)
# print(layer_output.shape)

# np.save('./enrollment_data/speaker_model.npy', layer_output)


# layer_output = np.load('./enrollment_data/speaker_model.npy')
# average vector likewise d-vector

speaker_model_representation=np.empty(shape=[0,512])
m=0
for n in range(4):
	temp=np.mean(layer_output[m:m+15,:], axis=0)
	speaker_model_representation=np.vstack((speaker_model_representation,temp))
	m+=15

print(speaker_model_representation)
print(speaker_model_representation.shape)
# input()

np.save('./enrollment_data/speaker_model.npy', speaker_model_representation)


def plot_history(history, result_dir):
	plt.plot(history.history['acc'], marker='.')
	plt.title('model accuracy')
	plt.xlabel('epoch')
	plt.ylabel('accuracy')
	plt.grid()
	plt.legend(['acc'], loc='lower right')
	plt.show()
	plt.savefig(os.path.join(result_dir, './enrollment_data/enrollment_accuracy.png'))
	plt.close()

	plt.plot(history.history['loss'], marker='.')
	plt.title('model loss')
	plt.xlabel('epoch')
	plt.ylabel('loss')
	plt.grid()
	plt.legend(['loss'], loc='upper right')
	plt.show()
	plt.savefig(os.path.join(result_dir, './enrollment_data/enrollment_loss.png'))
	plt.close()

# plot_history(history,os.path.dirname(os.path.abspath(__file__)))