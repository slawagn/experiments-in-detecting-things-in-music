from time import perf_counter
import essentia.standard as estd
from essentia.pytools.spectral import hpcpgram
import matplotlib.pyplot as plt
import os
import shutil
import matplotlib
matplotlib.use('Agg')

start_time = None


def start_benchmark():
    global start_time
    start_time = perf_counter()


def stop_benchmark(msg):
    end_time = perf_counter()
    execution_time = end_time - start_time
    print(f"{msg}: {execution_time:.6f}s")


start_benchmark()
audio = estd.MonoLoader(
    filename='assets/2026-07-20-1-shamisen-thing-recording-setup-test_freeze_Shamisen 1.wav', sampleRate=11025)()
stop_benchmark("loaded audio")

start_benchmark()
ref_hpcp = hpcpgram(audio, sampleRate=11025)
stop_benchmark("build hpcp")

print(ref_hpcp)
print(ref_hpcp.shape)

query_hpcp = ref_hpcp[45:70]

shutil.rmtree('plots')
os.makedirs('plots', exist_ok=True)

fig = plt.gcf()
fig.set_size_inches(50.0, 4.5)
plt.imshow(ref_hpcp.T, aspect='auto', origin='lower', interpolation='none')
plt.title("HPCP stem")
plt.savefig('plots/full performance hpcp.png')

fig = plt.gcf()
fig.set_size_inches(50.0, 4.5)
plt.imshow(query_hpcp.T, aspect='auto', origin='lower', interpolation='none')
plt.title("HPCP stem")
plt.savefig('plots/query hpcp.png')

start_benchmark()
same_crp = estd.ChromaCrossSimilarity(frameStackSize=9,
                                      frameStackStride=1,
                                      binarizePercentile=0.095,
                                      oti=True)
same_pair_crp = same_crp(query_hpcp, ref_hpcp)

same_pair_score_matrix, same_pair_distance = estd.CoverSongSimilarity(disOnset=0.5,
                                                                      disExtension=0.5,
                                                                      alignmentType='serra09',
                                                                      distanceType='asymmetric')(same_pair_crp)
stop_benchmark("analyzed")


fig = plt.gcf()
fig.set_size_inches(15.5, 5.5)
plt.title('sameity distance: %s' % same_pair_distance)
plt.xlabel('Query')
plt.ylabel('Reference')
plt.imshow(same_pair_score_matrix, origin='lower')
plt.savefig('plots/same similarity.png')

print(same_pair_score_matrix)
print(same_pair_score_matrix.shape)
