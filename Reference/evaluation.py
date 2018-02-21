from keras.models import load_model
from keras.utils.vis_utils import plot_model
import scipy.io.wavfile as wav
import numpy as np
import os
lib_path = os.path.abspath(os.path.join('..'))
print(lib_path)
sys.path.append(lib_path)
import speechpy
import matplotlib.pyplot as plt
import keras
from sklearn.metrics.pairwise import cosine_similarity


s1_model = np.load('speaker1_model.npy')
print(s1_model)
print(s1_model.shape)
print(type(s1_model))

s2_model = np.load('speaker2_model.npy')
print(s2_model)
print(s2_model.shape)
print(type(s2_model))

s3_model = np.load('speaker3_model.npy')
print(s3_model)
print(s3_model.shape)
print(type(s3_model))


s4_model = np.load('speaker4_model.npy')
print(s4_model)
print(s4_model.shape)
print(type(s4_model))


s5_model = np.load('speaker5_model.npy')
print(s5_model)
print(s5_model.shape)
print(type(s5_model))


s6_model = np.load('speaker6_model.npy')
print(s6_model)
print(s6_model.shape)
print(type(s6_model))

background_model = load_model('my_model_prelu_1.h5')
background_model.summary()


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

x_test=all_mfec

x_test = np.transpose(x_test[None, :, :, :, :], axes=(1, 4, 2, 3, 0))

y_test=np.zeros(1)
y_test[0]=5

num_classes=6
y_test = keras.utils.to_categorical(y_test, num_classes)

history = background_model.fit(x_test, y_test, validation_data=None, batch_size=20, epochs=10, verbose=1, shuffle=True)

from keras import backend as K
# with a Sequential model

get_15th_layer_output = K.function([background_model.layers[0].input, K.learning_phase()],
                                  [background_model.layers[15].output])


layer_output = get_15th_layer_output([x_test, 1])[0]
print(layer_output)
print(layer_output.shape)

score_1 = cosine_similarity(s1_model, layer_output)
print(score_1)
score_2 = cosine_similarity(s2_model, layer_output)
print(score_2)
score_3 = cosine_similarity(s3_model, layer_output)
print(score_3)
score_4 = cosine_similarity(s4_model, layer_output)
print(score_4)
score_5 = cosine_similarity(s5_model, layer_output)
print(score_5)
score_6 = cosine_similarity(s6_model, layer_output)
print(score_6)

score_vector = np.zeros((6, 1))
target_label_vector = np.zeros((6, 1))

score_vector[0]=score_1
score_vector[1]=score_2
score_vector[2]=score_3
score_vector[3]=score_4
score_vector[4]=score_5
score_vector[5]=score_6

target_label_vector[0]=0
target_label_vector[1]=0
target_label_vector[2]=0
target_label_vector[3]=0
target_label_vector[4]=0
target_label_vector[5]=1

np.save('score_vector.npy', score_vector)
np.save('target_label_vector.npy', target_label_vector)
