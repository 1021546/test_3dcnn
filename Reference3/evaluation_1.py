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
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

# Keras
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
config.gpu_options.allocator_type = 'BFC'
set_session(tf.Session(config=config))

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(sess)


speaker_model = np.load('./enrollment_data/speaker_model.npy')
# print(speaker_model)
# print(speaker_model.shape)
# print(type(speaker_model))
background_model = load_model('./development_data/my_development_model_1.h5')
# background_model.summary()


# evaluation data
# temp_frame=1000
# temp_i=-1
# temp_j=-1

all_speaker_mfec=np.empty(shape=[0,20,99,40])


for i in range(1,5):
	one_speaker_mfec=np.empty(shape=[0,20,99,40])
	more_evaluation_mfec=np.empty(shape=[0,99,40])
	for j in range(0,25):
		repeat_evaluation_mfec=np.empty(shape=[0,99,40])
		one_evaluation_mfec=np.empty(shape=[0,40])
		file_path = './evaluation_wav_3/'+str(i)+'/'+str(j)+'.wav'
		file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_path)
		fs, signal = wav.read(file_name)

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

		one_evaluation_mfec=np.vstack((one_evaluation_mfec,logenergy[0:99]))

		# decide minimum frame
		# if(temp_frame>(logenergy.shape[0])):
		# 	temp_frame=logenergy.shape[0]
		# 	temp_i=i
		# 	temp_j=j
		repeat_evaluation_mfec = np.reshape(one_evaluation_mfec, (1, 99, 40))
		repeat_evaluation_mfec = np.repeat(repeat_evaluation_mfec,20,axis=0)
		# print(repeat_evaluation_mfec)
		# print(repeat_evaluation_mfec.shape)
		more_evaluation_mfec=np.vstack((more_evaluation_mfec,repeat_evaluation_mfec))
	# print(more_evaluation_mfec)
	# print(more_evaluation_mfec.shape)
	# print(type(more_evaluation_mfec))
	# input()
	one_speaker_mfec=np.reshape(more_evaluation_mfec, (25 ,20, 99, 40))
	# print(one_speaker_mfec)
	# print(one_speaker_mfec.shape)
	# print(type(one_speaker_mfec))
	# input()
	all_speaker_mfec=np.vstack((all_speaker_mfec,one_speaker_mfec))
	# print(all_speaker_mfec)
	# print(all_speaker_mfec.shape)
	# print(type(all_speaker_mfec))
	# input()

# print(temp_frame)
# print(temp_i)
# print(temp_j)

# print(all_speaker_mfec)
# print(all_speaker_mfec.shape)
# print(type(all_speaker_mfec))
# input()

x_test = all_speaker_mfec
np.save('./evaluation_data/x_test.npy', x_test)
# x_test = np.load('./evaluation_data/x_test.npy')

# print(x_test)
# print(x_test.shape)
# print(type(x_test))
# input()

y_test=np.zeros(100, dtype=np.int)
for i in range(0,100):
	label=i//25
	y_test[i]=label

np.save('./evaluation_data/y_test.npy', y_test)
# y_test = np.load('./evaluation_data/y_test.npy')

# print(y_test)
# print(y_test.shape)
# print(type(y_test))
# input()

# num_classes = 5
# batch_size = 128
# epochs = 20

# y_test = keras.utils.to_categorical(y_test, num_classes)

# history = background_model.fit(x_test, y_test, validation_data=None, batch_size=batch_size, 
# 	epochs=epochs, verbose=1, shuffle=True)


get_26th_layer_output = K.function([background_model.layers[0].input, K.learning_phase()],
                                  [background_model.layers[26].output])


layer_output = get_26th_layer_output([x_test, 1])[0]
# print(layer_output)
# print(layer_output.shape)
# input()

np.save('./evaluation_data/test_speaker.npy', layer_output)
# layer_output = np.load('./evaluation_data/test_speaker.npy')
# print(layer_output)
# print(layer_output.shape)

def plot_history(history, result_dir):
	plt.plot(history.history['acc'], marker='.')
	plt.title('model accuracy')
	plt.xlabel('epoch')
	plt.ylabel('accuracy')
	plt.grid()
	plt.legend(['acc'], loc='lower right')
	plt.show()
	plt.savefig(os.path.join(result_dir, './evaluation_data/evaluation_accuracy.png'))
	plt.close()

	plt.plot(history.history['loss'], marker='.')
	plt.title('model loss')
	plt.xlabel('epoch')
	plt.ylabel('loss')
	plt.grid()
	plt.legend(['loss'], loc='upper right')
	plt.show()
	plt.savefig(os.path.join(result_dir, './evaluation_data/evaluation_loss.png'))
	plt.close()

# plot_history(history,os.path.dirname(os.path.abspath(__file__)))

score_vector = np.empty(shape=[0,1])
target_label_vector = np.empty(shape=[0,1])

# print(layer_output.shape)
# print(speaker_model.shape)

for n in range(0,(layer_output.shape)[0]):
	for m in range(0,(speaker_model.shape)[0]):
		score = cosine_similarity(layer_output[n:n+1,:], speaker_model[m:m+1,:])
		# print("score: ", score)
		# print(score.shape)
		# print(type(score))
		# input()
		score_vector=np.vstack((score_vector,score))
		if (n//2) == m:
			target_label_vector=np.vstack((target_label_vector,[1]))
		else:
			target_label_vector=np.vstack((target_label_vector,[0]))

# print(score_vector)
# print(score_vector.shape)
# print(type(score_vector))
# input()

# print(target_label_vector)
# print(target_label_vector.shape)
# print(type(target_label_vector))

np.save('./evaluation_data/score_vector.npy', score_vector)
# score_vector = np.load('./evaluation_data/score_vector.npy')
np.save('./evaluation_data/target_label_vector.npy', target_label_vector)
# target_label_vector = np.load('./evaluation_data/target_label_vector.npy')

# print(score_vector)
# print(score_vector.shape)
# print(type(score_vector))

# prediction
# # pred = background_model.predict_classes(x_test, batch_size, verbose=1)
pred=np.zeros((y_test.shape)[0], dtype=np.int)


# # print(pred)
# # print(pred.shape)
# # input()

# j=0
# for i in range((y_test.shape)[0]):
# 	temp=[score_vector[j,0],score_vector[j+1,0],score_vector[j+2,0],score_vector[j+3,0]]
# 	pred[i]=temp.index(max(temp))
# 	j+=4


for i in range((y_test.shape)[0]):
	maximum=0
	max_index=-1
	for j in range((speaker_model.shape)[0]):
		if score_vector[i*((speaker_model.shape)[0])+j,0]>maximum:
			maximum=score_vector[i*((speaker_model.shape)[0])+j,0]
			max_index=j
	# print(maximum)
	# print(max_index)
	# input()
	pred[i]=max_index

# print(pred)
# print(pred.shape)
# input()

print(pred)
# print(np.argmax(y_test,axis=1))
print(y_test)

from sklearn.metrics import classification_report,confusion_matrix

target_names=['Class 1','Class 2','Class 3','Class 4']
# print(classification_report(np.argmax(y_test,axis=1),pred,target_names=target_names))
# print(confusion_matrix(np.argmax(y_test,axis=1),pred))
print(classification_report(y_test,pred,target_names=target_names))
print(confusion_matrix(y_test,pred))