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
HOUGH_THRESHOLD = 300
THEME_START_WINDOW = 5.5 * FRAMES_PER_SECOND

audio = estd.MonoLoader(
    filename='assets/A Hollowed Skull.mp3', sampleRate=48000)()
# audio = estd.MonoLoader(
#     filename='assets/Saltatio Favillae.mp3', sampleRate=48000)()
# audio = estd.MonoLoader(
#     filename='assets/Polumnia Omnia.mp3', sampleRate=48000)()

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

fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].imshow(self_crp, origin='lower')
ax[0].secondary_xaxis('top',
                      functions=(
                          lambda f: f / FRAMES_PER_SECOND,
                          lambda s: s * FRAMES_PER_SECOND))
ax[0].secondary_yaxis('right',
                      functions=(
                          lambda f: f / FRAMES_PER_SECOND,
                          lambda s: s * FRAMES_PER_SECOND))
ax[0].set_xlabel('y')
ax[0].set_ylabel('x')
ax[1].secondary_xaxis('top',
                      functions=(
                          lambda f: f / FRAMES_PER_SECOND,
                          lambda s: s * FRAMES_PER_SECOND))
ax[1].set_xlabel('y')

groups = defaultdict(lambda: defaultdict(list))
current_group = []

lines = sorted(lines, key=lambda line: line[1])
avg_start_x_of_group = 0
for line in lines:
    y1, x1, y2, x2 = line
    angle = math.atan2(y2 - y1, x2 - x1)
    if not (math.pi / 4 - math.pi / 32 <= angle <= math.pi / 4 + math.pi / 32):
        continue

    if not current_group:
        current_group.append(line)
        continue
    prev_avg_start_x_of_group = avg_start_x_of_group
    avg_start_x_of_group = sum(
        [l[1] for l in current_group]) / len(current_group)
    if abs(avg_start_x_of_group - x1) > THEME_START_WINDOW:
        i = int(prev_avg_start_x_of_group)
        groups[i] = {
            'start_x': round(sum([l[1] for l in current_group]) / len(current_group)),
            'end_x': round(sum([l[3] for l in current_group]) / len(current_group)),
            'lines': current_group
        }
        groups[i]['center_x'] = round(
            groups[i]['start_x'] + groups[i]['end_x']) / 2
        current_group = [line]
    else:
        current_group.append(line)
first_fragment = {
    'start_x': 0,
    'end_x': list(groups.keys())[0],
    'center_x': round(list(groups.keys())[0] / 2),
    'lines': []
}
groups[0] = first_fragment

# trim group ends to start of next group
keys = sorted(groups.keys())
for i in range(len(keys)):
    if i >= len(keys) - 1:
        break
    start = keys[i]
    groups[start]['end_x'] = min(
        groups[start]['end_x'], groups[keys[i+1]]['start_x'])

# split on all group x starts
keys = sorted(groups.keys())
lines_to_append_here = []
for i in range(len(keys)):
    start = keys[i]
    lines_to_remove = []
    lines_to_append = []
    for line in groups[start]['lines']:
        y1, x1, y2, x2 = line
        for cutoff in keys:
            if x1 < cutoff < x2:
                split_x = int(cutoff)
                slope = (y2 - y1) / (x2 - x1)
                split_y = round((split_x - x1) * slope + y1)
                segment1 = np.array([y1, x1, split_y, split_x], dtype='int32')
                segment2 = np.array([split_y, split_x, y2, x2], dtype='int32')
                lines_to_remove.append(line)
                lines_to_append += [segment1, segment2]
    for l in lines_to_append:
        key = start
        for k in keys:
            if abs(l[1] - k) <= 50:
                key = k
        groups[key]['lines'].append(l)
    for rmline in lines_to_remove:
        groups[start]['lines'] = [l for l in groups[start]
                                  ['lines'] if not np.array_equal(l, rmline)]

# split on all group x ends
lines_to_append_here = []
for i in range(len(keys)):
    start = keys[i]
    lines_to_remove = []
    lines_to_append = []
    for line in groups[start]['lines']:
        y1, x1, y2, x2 = line
        for k in keys:
            cutoff = groups[k]['end_x']
            if x1 < cutoff < x2:
                split_x = int(cutoff)
                slope = (y2 - y1) / (x2 - x1)
                split_y = round((split_x - x1) * slope + y1)
                segment1 = np.array([y1, x1, split_y, split_x], dtype='int32')
                segment2 = np.array([split_y, split_x, y2, x2], dtype='int32')
                lines_to_remove.append(line)
                lines_to_append += [segment1, segment2]
    for l in lines_to_append:
        key = start
        for k in keys:
            if abs(l[1] - k) <= 50:
                key = k
        groups[key]['lines'].append(l)
    for rmline in lines_to_remove:
        groups[start]['lines'] = [l for l in groups[start]
                                  ['lines'] if not np.array_equal(l, rmline)]

# split on all group y starts
lines_to_append_here = []
for i in range(len(keys)):
    start = keys[i]
    lines_to_remove = []
    lines_to_append = []
    for line in groups[start]['lines']:
        y1, x1, y2, x2 = line
        for cutoff in keys:
            if y1 < cutoff < y2:
                split_y = int(cutoff)
                slope = (y2 - y1) / (x2 - x1)
                split_x = round((split_y - y1) / slope + x1)
                segment1 = np.array([y1, x1, split_y, split_x], dtype='int32')
                segment2 = np.array([split_y, split_x, y2, x2], dtype='int32')
                lines_to_remove.append(line)
                lines_to_append += [segment1, segment2]
    for l in lines_to_append:
        key = start
        for k in keys:
            if abs(l[1] - k) <= 50:
                key = k
        groups[key]['lines'].append(l)
    for rmline in lines_to_remove:
        groups[start]['lines'] = [l for l in groups[start]
                                  ['lines'] if not np.array_equal(l, rmline)]

# split on all group y ends
lines_to_append_here = []
for i in range(len(keys)):
    start = keys[i]
    lines_to_remove = []
    lines_to_append = []
    for line in groups[start]['lines']:
        y1, x1, y2, x2 = line
        for k in keys:
            cutoff = groups[k]['end_x']
            if y1 < cutoff < y2:
                split_y = int(cutoff)
                slope = (y2 - y1) / (x2 - x1)
                split_x = round((split_y - y1) / slope + x1)
                segment1 = np.array([y1, x1, split_y, split_x], dtype='int32')
                segment2 = np.array([split_y, split_x, y2, x2], dtype='int32')
                lines_to_remove.append(line)
                lines_to_append += [segment1, segment2]
    for l in lines_to_append:
        key = start
        for k in keys:
            if abs(l[1] - k) <= 50:
                key = k
        groups[key]['lines'].append(l)
    for rmline in lines_to_remove:
        groups[start]['lines'] = [l for l in groups[start]
                                  ['lines'] if not np.array_equal(l, rmline)]


# prune short fragments
for k in keys:
    lines_to_remove = []
    for line in groups[k]['lines']:
        y1, x1, y2, x2 = line
        if math.sqrt((y2 - y1)**2 + (x2 - x1)**2) >= MIN_MOTIF_LENGTH:
            continue
        lines_to_remove.append(line)
    for rmline in lines_to_remove:
        groups[k]['lines'] = [l for l in groups[k]
                              ['lines'] if not np.array_equal(l, rmline)]

keys = sorted(list(set(list(groups.keys()) + [groups[start]['end_x']
                                              for start in groups.keys()])))

# find matrix of restatement starts
restatements = defaultdict(dict)
for i in range(1, len(keys)):
    y = keys[i]
    for j in range(i):
        x = keys[j]
        for line in groups[y]['lines']:
            if abs(line[1] - y) <= 50 and abs(line[0] - x <= 50):
                restatements[y][x] = True
                break

for y in restatements.keys():
    current_occurrence = [y, groups[y]['end_x']]
    color = random.choice(list(mcolors.CSS4_COLORS.keys()))
    ax[0].plot([current_occurrence[0], current_occurrence[1]],
               [0, 100], color=color, linewidth=10)
    ax[1].plot([current_occurrence[0], current_occurrence[1]],
               [0, 1], color=color, linewidth=10)
    for x in restatements[y].keys():
        # this is hacky, but so is the rest of the code here...
        if not (groups[x]['end_x'] and groups[y]['end_x']):
            continue
        previous_occurrence = [x, groups[x]['end_x']]
        ax[0].plot([previous_occurrence[0], previous_occurrence[1]],
                   [0, 100], color=color, linewidth=10)
        ax[1].plot([previous_occurrence[0], previous_occurrence[1]],
                   [0, 1], color=color, linewidth=10)


random.seed(47)
for start in groups.keys():
    color = random.choice(list(mcolors.CSS4_COLORS.keys()))
    ax[0].plot([0, self_crp.shape[1]], [groups[start]['start_x'],
                                        groups[start]['start_x']], color=color)
    ax[0].plot([groups[start]['start_x'], groups[start]['start_x']],
               [0, self_crp.shape[1]], color=color)
    ax[0].plot([0, self_crp.shape[1]], [groups[start]['end_x'],
                                        groups[start]['end_x']], color=color)
    ax[0].plot([groups[start]['end_x'], groups[start]['end_x']],
               [0, self_crp.shape[1]], color=color)
    for line in groups[start]['lines']:
        y1, x1, y2, x2 = line
        ax[0].plot([y1, y2,], [x1, x2], color=color)
        ax[0].scatter([y1, y2,], [x1, x2], color='red', s=10, zorder=2.02)
plt.show()
