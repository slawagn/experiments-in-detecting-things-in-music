import os
import matplotlib.pyplot as plt
from essentia.pytools.spectral import hpcpgram
import essentia.standard as estd
from time import perf_counter
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
    filename='assets/optimized-in-an-improper-way/2026-07-20-1-shamisen-thing-recording-setup-test_freeze_Shamisen 1.wav', sampleRate=11025)()
stop_benchmark("loaded audio")

start_benchmark()
ref_hpcp = hpcpgram(audio, sampleRate=11025)
stop_benchmark("built hpcp")

query_hpcp = ref_hpcp[20:55]

shutil.rmtree('plots')
os.makedirs('plots', exist_ok=True)

start_benchmark()
crp = estd.ChromaCrossSimilarity(frameStackSize=9,
                                 frameStackStride=1,
                                 binarizePercentile=0.095,
                                 oti=True)
stop_benchmark("initialized crp")

start_benchmark()
STEP_SIZE = 5
WINDOW_SIZE = 40
results = []
for i in range(0, len(ref_hpcp) - WINDOW_SIZE, STEP_SIZE):
    pair_crp = crp(query_hpcp, ref_hpcp[i:i + WINDOW_SIZE])
    score_matrix, pair_distance = estd.CoverSongSimilarity(disOnset=0.5,
                                                           disExtension=0.5,
                                                           alignmentType='serra09',
                                                           distanceType='asymmetric')(pair_crp)
    results.append(((i, i + 200), pair_distance, score_matrix))
stop_benchmark("analyzed")

for result in results:
    print(f"{result[0][0]}-{result[0][1]}: {result[1]}")


start_benchmark()
for result in results:
    fig = plt.gcf()
    plt.title('sameity distance: %s' % result[1])
    plt.xlabel('Query')
    plt.ylabel('Reference')
    plt.imshow(result[2], origin='lower')
    plt.savefig(f"plots/{result[0][0]}-{result[0][1]}.png")

fig = plt.gcf()
plt.imshow(ref_hpcp.T, aspect='auto', origin='lower', interpolation='none')
plt.title("HPCP stem")
plt.savefig('plots/ref hpcp.png')

fig = plt.gcf()
plt.imshow(query_hpcp.T, aspect='auto', origin='lower', interpolation='none')
plt.title("HPCP stem")
plt.savefig('plots/query hpcp.png')
stop_benchmark("saved plots")
