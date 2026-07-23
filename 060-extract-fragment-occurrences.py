from time import perf_counter
import essentia.standard as estd
from essentia.pytools.spectral import hpcpgram
import cv2
import numpy as np
from scipy.signal import correlate
import matplotlib.pyplot as plt
import os
import shutil
import matplotlib


audio = estd.MonoLoader(
    filename='assets/Chrysalis Suspirii.mp3', sampleRate=48000)()

ref_hpcp = hpcpgram(audio, sampleRate=48000)

query_hpcp = ref_hpcp[700:1200]
# ref_hpcp = ref_hpcp[1300:]

# shutil.rmtree('plots')
# os.makedirs('plots', exist_ok=True)

# fig = plt.gcf()
# fig.set_size_inches(50.0, 4.5)
# plt.imshow(ref_hpcp.T, aspect='auto', origin='lower', interpolation='none')
# plt.title("Track HPCP")
# plt.savefig('plots/track hpcp.png')


crp = estd.ChromaCrossSimilarity(frameStackSize=9,
                                 frameStackStride=1,
                                 binarizePercentile=0.095,
                                 oti=True)
self_crp = crp(query_hpcp, ref_hpcp)

kernel = np.eye(self_crp.shape[0])
corr = correlate(self_crp, kernel)

corr = np.uint8(corr)
corr = np.where(corr < 112, 0, corr)
corr = cv2.GaussianBlur(corr, (127, 127), 0)
lines = cv2.HoughLinesP(corr, rho=1, theta=np.pi/180,
                        threshold=100, minLineLength=50, maxLineGap=10)

fig = plt.gcf()
plt.imshow(corr, origin='lower')
for pair in lines:
    x1, y1, x2, y2 = pair
    plt.plot([x1, x2], [10, 10], color='red')
plt.show()
