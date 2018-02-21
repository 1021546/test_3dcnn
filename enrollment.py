from keras.models import load_model
from keras.utils.vis_utils import plot_model
import scipy.io.wavfile as wav
import numpy as np
import os
import sys
lib_path = os.path.abspath(os.path.join('..'))
print(lib_path)
sys.path.append(lib_path)
import speechpy
import matplotlib.pyplot as plt
import keras

background_model = load_model('my_model_prelu_1.h5')

background_model.summary()
# plot_model(background_model, show_shapes=True, to_file='background_model.png')

all_mfec=np.empty(shape=[0,40,30])
speaker_mfec=np.empty(shape=[0,40])
for i in range(0,6):
	for j in range(1,6):
		file_path = './wav/6/'+str(i)+'_'+str(j)+'.wav'
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

x_train=all_mfec

x_train = np.transpose(x_train[None, :, :, :, :], axes=(1, 4, 2, 3, 0))

y_train=np.zeros(1)
y_train[0]=5

num_classes=6
y_train = keras.utils.to_categorical(y_train, num_classes)

history = background_model.fit(x_train, y_train, validation_data=None, batch_size=20, epochs=10, verbose=1, shuffle=True)

from keras import backend as K
# with a Sequential model

get_15th_layer_output = K.function([background_model.layers[0].input, K.learning_phase()],
                                  [background_model.layers[15].output])


layer_output = get_15th_layer_output([x_train, 1])[0]
print(layer_output)
print(layer_output.shape)

np.save('speaker6_model.npy', layer_output)
# x_test = np.load('speaker1_model.npy')
# print(x_test)
# print(x_test.shape)
# print(type(x_test))


def plot_history(history, result_dir):
	plt.plot(history.history['acc'], marker='.')
	plt.title('model accuracy')
	plt.xlabel('epoch')
	plt.ylabel('accuracy')
	plt.grid()
	plt.legend(['acc'], loc='lower right')
	plt.savefig(os.path.join(result_dir, 's6_model_accuracy.png'))
	plt.close()

	plt.plot(history.history['loss'], marker='.')
	plt.title('model loss')
	plt.xlabel('epoch')
	plt.ylabel('loss')
	plt.grid()
	plt.legend(['loss'], loc='upper right')
	plt.savefig(os.path.join(result_dir, 's6_model_loss.png'))
	plt.close()

# plot_history(history,os.path.dirname(os.path.abspath(__file__)))