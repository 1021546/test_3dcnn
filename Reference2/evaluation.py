import numpy as np
import scipy.io.wavfile as wav
import os
import sys
lib_path = os.path.abspath(os.path.join('..'))
print(lib_path)
sys.path.append(lib_path)
import speechpy
import keras
from keras.models import load_model
from keras import backend as K
# with a Sequential model
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


speaker_model = np.load('speaker_model.npy')
# print(speaker_model)
# print(speaker_model.shape)
# print(type(speaker_model))
background_model = load_model('my_development_model.h5')
background_model.summary()


# evaluation data
temp_frame=1000
temp_i=-1
temp_j=-1

all_speaker_mfec=np.empty(shape=[0,20,110,40])
one_speaker_mfec=np.empty(shape=[0,110,40])

for i in range(1,6):
	one_evaluation_mfec=np.empty(shape=[0,40])
	for j in range(0,20):
		file_path = './speaker_wav/'+str(i)+'/evaluation/'+str(j)+'.wav'
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

		one_evaluation_mfec=np.vstack((one_evaluation_mfec,logenergy[0:110]))

		# decide minimum frame
		# if(temp_frame>(logenergy.shape[0])):
		# 	temp_frame=logenergy.shape[0]
		# 	temp_i=i
		# 	temp_j=j
	one_evaluation_mfec = np.reshape(one_evaluation_mfec, (20, 110, 40))
	one_speaker_mfec=np.vstack((one_speaker_mfec,one_evaluation_mfec))

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
y_test=np.zeros(5)
for i in range(0,5):
	y_test[i]=i

# print(y_test)
# print(y_test.shape)
# print(type(y_test))
# input()

num_classes = 5
batch_size = 128
epochs = 20

y_test = keras.utils.to_categorical(y_test, num_classes)

history = background_model.fit(x_test, y_test, validation_data=None, batch_size=batch_size, 
	epochs=epochs, verbose=1, shuffle=True)


get_15th_layer_output = K.function([background_model.layers[0].input, K.learning_phase()],
                                  [background_model.layers[15].output])


layer_output = get_15th_layer_output([x_test, 1])[0]
# print(layer_output)
# print(layer_output.shape)

def plot_history(history, result_dir):
	plt.plot(history.history['acc'], marker='.')
	plt.title('model accuracy')
	plt.xlabel('epoch')
	plt.ylabel('accuracy')
	plt.grid()
	plt.legend(['acc'], loc='lower right')
	plt.savefig(os.path.join(result_dir, 'evaluation_accuracy.png'))
	plt.close()

	plt.plot(history.history['loss'], marker='.')
	plt.title('model loss')
	plt.xlabel('epoch')
	plt.ylabel('loss')
	plt.grid()
	plt.legend(['loss'], loc='upper right')
	plt.savefig(os.path.join(result_dir, 'evaluation_loss.png'))
	plt.close()

# plot_history(history,os.path.dirname(os.path.abspath(__file__)))

score_vector = np.empty(shape=[0,1])
target_label_vector = np.empty(shape=[0,1])

for n in range(0,(layer_output.shape)[0]):
	for m in range(0,(speaker_model.shape)[0]):
		score = cosine_similarity(layer_output[n:n+1,:], speaker_model[m:m+1,:])
		# print("score: ", score)
		# print(score.shape)
		# print(type(score))
		# input()
		score_vector=np.vstack((score_vector,score))
		if n == m:
			target_label_vector=np.vstack((target_label_vector,[1]))
		else:
			target_label_vector=np.vstack((target_label_vector,[0]))

print(score_vector)
print(score_vector.shape)
print(type(score_vector))

print(target_label_vector)
print(target_label_vector.shape)
print(type(target_label_vector))

np.save('score_vector.npy', score_vector)
np.save('target_label_vector.npy', target_label_vector)