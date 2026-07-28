import numpy as np
import essentia.standard as estd
from essentia.pytools.spectral import hpcpgram
import cv2
import math
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pprint import pprint

AUDIOBUF_WINDOW_LEN = 48000  # samples
HOPSIZE = 2000  # samples per frame
FRAMES_PER_SECOND = AUDIOBUF_WINDOW_LEN / HOPSIZE
MIN_MOTIF_LENGTH = 5 * FRAMES_PER_SECOND  # frames
MAX_LINE_GAP = 2 * FRAMES_PER_SECOND
HOUGH_THRESHOLD = 320
THEME_START_WINDOW = 5.5 * FRAMES_PER_SECOND
THEME_END_WINDOW = 5 * FRAMES_PER_SECOND

audio = estd.MonoLoader(
    filename='assets/A Hollowed Skull.mp3', sampleRate=48000)()

ref_hpcp = hpcpgram(audio, sampleRate=48000, hopSize=HOPSIZE)
query_hpcp = ref_hpcp

crp_analyzer = estd.ChromaCrossSimilarity(frameStackSize=9,
                                          frameStackStride=1,
                                          binarizePercentile=0.095,
                                          oti=True)

self_crp = crp_analyzer(query_hpcp, ref_hpcp)
self_crp = np.uint8(self_crp)

y, x = np.indices(self_crp.shape)
self_crp = np.where(y <= x + MIN_MOTIF_LENGTH, 0, self_crp)
# self_crp = cv2.GaussianBlur(self_crp, (3, 3), 0)

lines = cv2.HoughLinesP(self_crp, rho=1, theta=np.pi/180,
                        threshold=HOUGH_THRESHOLD, minLineLength=MIN_MOTIF_LENGTH, maxLineGap=MAX_LINE_GAP)

fig, ax = plt.subplots()
plt.imshow(self_crp, origin='lower')
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
ax.secondary_xaxis('bottom',
                   functions=(
                       lambda f: f / FRAMES_PER_SECOND,
                       lambda s: s * FRAMES_PER_SECOND))
ax.secondary_yaxis('left',
                   functions=(
                       lambda f: f / FRAMES_PER_SECOND,
                       lambda s: s * FRAMES_PER_SECOND))

groups = dict()
current_group = []

lines = sorted(lines, key=lambda line: line[1])
avg_start_y_of_group = 0
for line in lines:
    y1, x1, y2, x2 = line
    angle = math.atan2(y2 - y1, x2 - x1)
    if not (math.pi / 4 - math.pi / 32 <= angle <= math.pi / 4 + math.pi / 32):
        continue

    if not current_group:
        current_group.append(line)
        continue
    prev_avg_start_y_of_group = avg_start_y_of_group
    avg_start_y_of_group = sum(
        [l[1] for l in current_group]) / len(current_group)
    if abs(avg_start_y_of_group - x1) > THEME_START_WINDOW:
        i = int(prev_avg_start_y_of_group)
        groups[i] = {
            'start_y': round(sum([l[1] for l in current_group]) / len(current_group)),
            'end_y': round(sum([l[3] for l in current_group]) / len(current_group)),
            'lines': current_group
        }
        groups[i]['center_y'] = round(
            groups[i]['start_y'] + groups[i]['end_y']) / 2
        current_group = [line]
    else:
        current_group.append(line)

# trim group ends to start of next group
keys = sorted(groups.keys())
for i in range(len(keys)):
    if i >= len(keys) - 1:
        break
    start = keys[i]
    groups[start]['end_y'] = min(
        groups[start]['end_y'], groups[keys[i+1]]['start_y'])

# split on all group starts
keys = sorted(groups.keys())
lines_to_append_here = []
for i in range(len(keys)):
    start = keys[i]
    lines_to_remove = []
    lines_to_append = []
    # if i >= len(keys):
    #     break
    for line in groups[start]['lines']:
        y1, x1, y2, x2 = line
        for cutoff in keys:
            if y1 < cutoff < y2:
                split_y = int(cutoff)
                slope = (y2 - y1) / (x2 - x1)
                split_x = round((split_y - y1) / slope) + x1
                segment1 = np.array([y1, x1, split_y, split_x], dtype='int32')
                segment2 = np.array([split_y, split_x, y2, x2], dtype='int32')
                lines_to_remove.append(line)
                lines_to_append += [segment1, segment2]
    for l in lines_to_append:
        key = start
        for k in keys:
            if abs(l[0] - k) <= 5:
                key = k
        # key = l[0] if l[0] in keys else start
        groups[key]['lines'].append(l)
    for rmline in lines_to_remove:
        groups[start]['lines'] = [l for l in groups[start]
                                  ['lines'] if not np.array_equal(l, rmline)]

# # split on group starts
# keys = sorted(groups.keys())
# lines_to_append_here = []
# for i in range(1, len(keys)):
#     start = keys[i]
#     lines_to_remove = []
#     lines_to_append_here = []
#     # if i >= len(keys):
#     #     break
#     for line in groups[start]['lines']:
#         y1, x1, y2, x2 = line
#         cutoff_y = groups[keys[i-1]]['start_y']
#         # cutoff_y = min(groups[start]['start_y'], groups[keys[i+1]]['start_y'])
#         # cutoff_y = groups[keys[i+1]]['end_y']
#         if y1 < cutoff_y:
#             split_y = int(cutoff_y)
#             slope = (y2 - y1) / (x2 - x1)
#             split_x = int((split_y - y1) / slope) + x1
#             # ax.scatter(split_y, split_x, color='red', s=50, marker='x')
#             segment1 = np.array([y1, x1, split_y, split_x], dtype='int32')
#             segment2 = np.array([split_y, split_x, y2, x2], dtype='int32')
#             lines_to_remove.append(line)
#             lines_to_append_here += [segment1, segment2]
#     for l in lines_to_append_here:
#         groups[start]['lines'].append(l)
#     for rmline in lines_to_remove:
#         groups[start]['lines'] = [l for l in groups[start]
#                                   ['lines'] if not np.array_equal(l, rmline)]


# # split on group ends
# keys = sorted(groups.keys())
# lines_to_append_to_next = []
# for i in range(len(keys)):
#     start = keys[i]
#     lines_to_remove = []
#     lines_to_append_to_next = []
#     if i >= len(keys) - 1:
#         break
#     for line in groups[start]['lines']:
#         y1, x1, y2, x2 = line
#         cutoff_y = groups[start]['start_y']
#         # cutoff_y = min(groups[start]['start_y'], groups[keys[i+1]]['start_y'])
#         # cutoff_y = groups[keys[i+1]]['end_y']
#         if y2 > cutoff_y:
#             split_y = int(cutoff_y)
#             slope = (y2 - y1) / (x2 - x1)
#             split_x = int((split_y - y1) / slope) + x1
#             # ax.scatter(split_y, split_x, color='red', s=50, marker='x')
#             segment1 = np.array([y1, x1, split_y, split_x], dtype='int32')
#             segment2 = np.array([split_y, split_x, y2, x2], dtype='int32')
#             lines_to_remove.append(line)
#             lines_to_append_to_next += [segment1, segment2]
#     for l in lines_to_append_to_next:
#         groups[keys[i+1]]['lines'].append(l)
#     for rmline in lines_to_remove:
#         groups[start]['lines'] = [l for l in groups[start]
#                                   ['lines'] if not np.array_equal(l, rmline)]

pprint(groups)
random.seed(47)
for start in groups.keys():
    color = random.choice(list(mcolors.CSS4_COLORS.keys()))
    ax.plot([0, self_crp.shape[1]], [groups[start]['start_y'],
            groups[start]['start_y']], color=color)
    ax.plot([groups[start]['start_y'], groups[start]['start_y']],
            [0, self_crp.shape[1]], color=color)
    ax.plot([0, self_crp.shape[1]], [groups[start]['end_y'],
            groups[start]['end_y']], color=color)
    ax.plot([groups[start]['end_y'], groups[start]['end_y']],
            [0, self_crp.shape[1]], color=color)
    for line in groups[start]['lines']:
        y1, x1, y2, x2 = line
        ax.plot([y1, y2,], [x1, x2], color=color)
        ax.scatter([y1, y2,], [x1, x2], color='red', s=10, zorder=2.02)

plt.show()
