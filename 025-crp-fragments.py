import os
import matplotlib.pyplot as plt
from essentia.pytools.spectral import hpcpgram
import essentia.standard as estd
from time import perf_counter
import shutil
import matplotlib
matplotlib.use('Agg')


audio = estd.MonoLoader(
    filename='assets/Chrysalis Suspirii.mp3', sampleRate=48000)()

ref_hpcp = hpcpgram(audio, sampleRate=48000)

query_hpcp = ref_hpcp[700:1200]

# shutil.rmtree('plots')
os.makedirs('plots', exist_ok=True)

crp = estd.ChromaCrossSimilarity(frameStackSize=9,
                                 frameStackStride=1,
                                 binarizePercentile=0.095,
                                 oti=True)

cross_similarity = crp(query_hpcp, ref_hpcp)

STEP_SIZE = 50
WINDOW_SIZE = 200
results = []
for i in range(0, cross_similarity.shape[1] - WINDOW_SIZE, STEP_SIZE):
    pair_crp = cross_similarity.T[i:i+WINDOW_SIZE, :]
    score_matrix, pair_distance = estd.CoverSongSimilarity(disOnset=0.5,
                                                           disExtension=0.5,
                                                           alignmentType='serra09',
                                                           distanceType='asymmetric')(pair_crp)
    results.append(((i, i + WINDOW_SIZE), pair_distance, score_matrix))

for result in results:
    print(f"{result[0][0]}-{result[0][1]}: {result[1]}")


# for result in results:
#     fig = plt.gcf()
#     plt.title('sameity distance: %s' % result[1])
#     plt.xlabel('Query')
#     plt.ylabel('Reference')
#     plt.imshow(result[2], origin='lower')
#     plt.savefig(f"plots/{result[0][0]}-{result[0][1]}.png")

# fig = plt.gcf()
# plt.imshow(ref_hpcp.T, aspect='auto', origin='lower', interpolation='none')
# plt.title("HPCP stem")
# plt.savefig('plots/ref hpcp.png')

# fig = plt.gcf()
# plt.imshow(query_hpcp.T, aspect='auto', origin='lower', interpolation='none')
# plt.title("HPCP stem")
# plt.savefig('plots/query hpcp.png')
