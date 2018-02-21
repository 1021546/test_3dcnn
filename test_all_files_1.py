import scipy.io.wavfile as wav
import numpy as np
import os
import sys
lib_path = os.path.abspath(os.path.join('..'))
print(lib_path)
sys.path.append(lib_path)
import speechpy
import os

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
	# input()
	all_mfec=np.vstack((all_mfec,speaker_mfec))
	# print(all_mfec)
	# print(all_mfec.shape)
	# print(type(all_mfec))
	# input()

# print(all_mfec_shape)
# print(all_mfec_shape.shape)
# print(np.amin(all_mfec_shape))

# print(speaker_mfec)
# print(speaker_mfec.shape)
# print(type(speaker_mfec))

all_mfec = np.reshape(all_mfec, (6, 99, 40, 30))

print(all_mfec)
print(all_mfec.shape)
print(type(all_mfec))


train_label=np.zeros(6)

for i in range(0,6):
	train_label[i]=i

print(train_label)
print(train_label.shape)
print(type(train_label))