from time import perf_counter
import essentia.standard as estd
from essentia.pytools.spectral import hpcpgram
import matplotlib.pyplot as plt
import os
import shutil
import matplotlib
matplotlib.use('Agg')


audio = estd.MonoLoader(
    filename='assets/Chrysalis Suspirii.mp3', sampleRate=48000)()

ref_hpcp = hpcpgram(audio, sampleRate=48000)

query_hpcp = ref_hpcp[700:1200]

shutil.rmtree('plots')
os.makedirs('plots', exist_ok=True)

fig = plt.gcf()
fig.set_size_inches(50.0, 4.5)
plt.imshow(ref_hpcp.T, aspect='auto', origin='lower', interpolation='none')
plt.title("Track HPCP")
plt.savefig('plots/track hpcp.png')


crp = estd.ChromaCrossSimilarity(frameStackSize=9,
                                 frameStackStride=1,
                                 binarizePercentile=0.095,
                                 oti=True)
self_crp = crp(query_hpcp, ref_hpcp)

same_pair_score_matrix, same_pair_distance = estd.CoverSongSimilarity(disOnset=0.5,
                                                                      disExtension=0.5,
                                                                      alignmentType='serra09',
                                                                      distanceType='asymmetric')(self_crp)


fig = plt.gcf()
fig.set_size_inches(15.5, 5.5)
plt.title('sameity distance: %s' % same_pair_distance)
plt.xlabel('Query')
plt.ylabel('Reference')
plt.imshow(self_crp, origin='lower')
plt.savefig('plots/same similarity.png')
