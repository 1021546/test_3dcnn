import wave
import numpy as np
import matplotlib.pyplot as plt
import Volume as vp

def findIndex(vol,thres):
    l = len(vol)
    ii = 0
    index=[]
    # index = np.zeros(6,dtype=np.int16)
    for i in range(l-1):
        if((vol[i]-thres)*(vol[i+1]-thres)<0):
        	index.append(i)
            # index[ii]=i
            # ii = ii+1
    # print(index)
    # print(len(index))
    # print(type(index))
    # input()

    if len(index)==0:
        index.append(0)
    return (index[0],index[-1])

fw = wave.open('./speaker_wav/1/train1/4.wav','r')
params = fw.getparams()
nchannels, sampwidth, framerate, nframes = params[:4]
strData = fw.readframes(nframes)
waveData = np.fromstring(strData, dtype=np.int16)
waveData = waveData*1.0/max(abs(waveData))  # normalization
fw.close()

frameSize = 256
overLap = 128
vol = vp.calVolume(waveData,frameSize,overLap)
threshold1 = max(vol)*0.10
threshold2 = min(vol)*10.0
threshold3 = max(vol)*0.05+min(vol)*5.0

time = np.arange(0,nframes) * (1.0/framerate)
frame = np.arange(0,len(vol)) * (nframes*1.0/len(vol)/framerate)


(start_index1,end_index1) = findIndex(vol,threshold1)
(start_index2,end_index2) = findIndex(vol,threshold2)
(start_index3,end_index3) = findIndex(vol,threshold3)

start_index1 = start_index1*(nframes*1.0/len(vol)/framerate)
end_index1 = end_index1*(nframes*1.0/len(vol)/framerate)
start_index2 = start_index2*(nframes*1.0/len(vol)/framerate)
end_index2 = end_index2*(nframes*1.0/len(vol)/framerate)
start_index3 = start_index3*(nframes*1.0/len(vol)/framerate)
end_index3 = end_index3*(nframes*1.0/len(vol)/framerate)
end = nframes * (1.0/framerate)


# print(nframes)
# print(time.shape)
# print(waveData.shape)

plt.subplot(211)
plt.plot(time,waveData,color="black")
plt.plot([start_index1,start_index1],[-1,1],'-r')
plt.plot([end_index1,end_index1],[-1,1],'-r')
plt.plot([start_index2,start_index2],[-1,1],'-g')
plt.plot([end_index2,end_index2],[-1,1],'-g')
plt.plot([start_index3,start_index3],[-1,1],'-b')
plt.plot([end_index3,end_index3],[-1,1],'-b')
plt.ylabel('Amplitude')

plt.subplot(212)
plt.plot(frame,vol,color="black")
plt.plot([0,end],[threshold1,threshold1],'-r', label="threshold 1")
plt.plot([0,end],[threshold2,threshold2],'-g', label="threshold 2")
plt.plot([0,end],[threshold3,threshold3],'-b', label="threshold 3")
plt.legend()
plt.ylabel('Volume(absSum)')
plt.xlabel('time(seconds)')
plt.show()


a=start_index3*framerate
b=end_index3*framerate

print(a)
print(b)
print(framerate)


import scipy.io.wavfile
scipy.io.wavfile.write("karplus.wav", framerate, waveData[int(a):int(b)])
rate, data = scipy.io.wavfile.read('karplus.wav')


import soundfile

data, samplerate = soundfile.read('karplus.wav')
soundfile.write('new.wav', data, samplerate, subtype='PCM_16')