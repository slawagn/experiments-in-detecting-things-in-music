# fmt: off
import matplotlib
matplotlib.use('TkAgg')

import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
import argparse
import queue
import sys
import random
from collections import defaultdict

import essentia.standard as estd
from essentia.pytools.spectral import hpcpgram
from scipy.signal import correlate
import cv2
from skimage.morphology import skeletonize
# fmt: on


AUDIOBUF_WINDOW_LEN = 48000  # samples
HOPSIZE = 2000  # samples per frame
FRAMES_PER_SECOND = AUDIOBUF_WINDOW_LEN / HOPSIZE
PHRASEBUF_LEN = 150  # frames
SIMILARITY_THRESHOLD = 70
MIN_MOTIF_LENGTH = 1.5 * FRAMES_PER_SECOND  # frames


class Motif:
    def __init__(self, x1, x2, motifClass):
        self._x1 = x1
        self._x2 = x2
        self._motifClass = motifClass

    @property
    def x1(self):
        return self._x1

    @property
    def x2(self):
        return self._x2

    @x2.setter
    def x2(self, value):
        self._x2 = value

    @property
    def motifClass(self):
        return self._motifClass


class MotifTracker:
    def __init__(self):
        self._motifs = dict()

    @property
    def motifs(self):
        return self._motifs

    def noticeSimilarities(self, previous_occurrence, current_occurrence):
        for existingMotif in self._motifs.values():
            # < existingMotif.x2:
            if previous_occurrence[0] < existingMotif.x1 < previous_occurrence[1]:
                print(
                    f"trimming new motifs to {previous_occurrence[0]} {existingMotif.x1}")
                previous_occurrence = (
                    previous_occurrence[0], existingMotif.x1)
                current_occurrence = (
                    current_occurrence[0], min(current_occurrence[1], current_occurrence[0] + previous_occurrence[1] - previous_occurrence[0]))
        if current_occurrence[0] < previous_occurrence[1]:
            previous_occurrence = (
                previous_occurrence[0], current_occurrence[0])
            current_occurrence = (
                current_occurrence[0], min(current_occurrence[1], current_occurrence[0] + previous_occurrence[1] - previous_occurrence[0]))

        newClass = max(
            [m.motifClass for m in self._motifs.values()] or [-1]) + 1
        newlyFoundFragments = []
        motifsToRemove = []

        if not self._motifs:
            newlyFoundFragments.append(previous_occurrence)
        else:
            exMotifsCopy = self._motifs.values()
            appending = True
            previousFragmentLength = previous_occurrence[1] - \
                previous_occurrence[0]
            for existingMotif in exMotifsCopy:
                if not appending:
                    break
                if existingMotif.x1 < previous_occurrence[0] < existingMotif.x2 < previous_occurrence[1]:
                    # print(
                    #     f"extending {existingMotif.x1} {existingMotif.x2} to {existingMotif.x1} {previous_occurrence[1]}")
                    existingMotif.x2 = previous_occurrence[1]
                    previousFragmentLength = min(previousFragmentLength, previous_occurrence[1] -
                                                 existingMotif.x1)
                    newClass = existingMotif.motifClass
                    appending = False
                elif previous_occurrence[0] <= existingMotif.x1 < existingMotif.x2 <= previous_occurrence[1]:
                    # print(
                    #     f"replacing {existingMotif.x1} {existingMotif.x2} with {previous_occurrence[0]} {previous_occurrence[1]}")
                    previousFragmentLength = min(previousFragmentLength, previous_occurrence[1] -
                                                 previous_occurrence[1])
                    newClass = existingMotif.motifClass
                    motifsToRemove.append(self._motifs[existingMotif.x1])
                elif existingMotif.x1 <= previous_occurrence[0] < previous_occurrence[1] <= existingMotif.x2:
                    # print(
                    #     f"skipping {previous_occurrence[0]} {previous_occurrence[1]}")
                    previousFragmentLength = min(
                        previousFragmentLength, existingMotif.x2 - existingMotif.x1)
                    newClass = existingMotif.motifClass
                    appending = False
                else:
                    previousFragmentLength = min(previousFragmentLength, previous_occurrence[1] -
                                                 previous_occurrence[0])
            if appending and previous_occurrence[1] - previous_occurrence[0] > MIN_MOTIF_LENGTH:
                # print(
                #     f"adding {previous_occurrence[0]} {previous_occurrence[1]}")
                newlyFoundFragments.append(previous_occurrence)

            print(
                f"new occurrence should not be longer than ({current_occurrence[0]}, {current_occurrence[0] + previousFragmentLength})")

            exMotifsCopy = self._motifs.values()
            appending = True
            for existingMotif in exMotifsCopy:
                if not appending:
                    break
                if existingMotif.x1 < current_occurrence[0] < existingMotif.x2 < current_occurrence[1]:
                    # print(
                    #     f"extending {existingMotif.x1} {existingMotif.x2} to {existingMotif.x1} {current_occurrence[1]}")
                    existingMotif.x2 = min(
                        current_occurrence[1], existingMotif.x1 + previousFragmentLength)
                    # newClass = existingMotif.motifClass
                    appending = False
                    # continue
                elif current_occurrence[0] <= existingMotif.x1 < existingMotif.x2 <= current_occurrence[1]:
                    # print(
                    #     f"replacing {existingMotif.x1} {existingMotif.x2} with {current_occurrence[0]} {current_occurrence[1]}")
                    newClass = existingMotif.motifClass
                    motifsToRemove.append(self._motifs[existingMotif.x1])
                    current_occurrence = (current_occurrence[0], min(
                        current_occurrence[1], current_occurrence[0] + previousFragmentLength))
                elif existingMotif.x1 <= current_occurrence[0] < current_occurrence[1] <= existingMotif.x2:
                    # print(
                    #     f"skipping {current_occurrence[0]} {current_occurrence[1]}")
                    # newClass = existingMotif.motifClass
                    appending = False
                else:
                    current_occurrence = (current_occurrence[0], min(
                        current_occurrence[1], current_occurrence[0] + previousFragmentLength))
                    # print(f"allowing {newFragment[0]} {newFragment[1]} so far")
            if appending and current_occurrence[1] - current_occurrence[0] > MIN_MOTIF_LENGTH:
                print(
                    f"adding {current_occurrence[0]} {current_occurrence[1]}")
                newlyFoundFragments.append(current_occurrence)

        for m in motifsToRemove:
            del self._motifs[m.x1]
        for f in newlyFoundFragments:
            self._motifs[f[0]] = Motif(f[0], f[1], newClass)


kernel = np.eye(PHRASEBUF_LEN)


def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    '-l', '--list-devices', action='store_true',
    help='show list of audio devices and exit')
args, remaining = parser.parse_known_args()
if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    parents=[parser])
parser.add_argument(
    'channels', type=int, default=[1], nargs='*', metavar='CHANNEL',
    help='input channels to plot (default: the first)')
parser.add_argument(
    '-d', '--device', type=int_or_str,
    help='input device (numeric ID or substring)')
parser.add_argument(
    '-w', '--window', type=float, default=200, metavar='DURATION',
    help='visible time slot (default: %(default)s ms)')
parser.add_argument(
    '-i', '--interval', type=float, default=100,
    help='minimum time between plot updates (default: %(default)s ms)')
parser.add_argument(
    '-b', '--blocksize', type=int, help='block size (in samples)')
parser.add_argument(
    '-r', '--samplerate', type=float, help='sampling rate of audio device')
parser.add_argument(
    '-n', '--downsample', type=int, default=1, metavar='N',
    help='display every Nth sample (default: %(default)s)')
args = parser.parse_args(remaining)
if any(c < 1 for c in args.channels):
    parser.error('argument CHANNEL: must be >= 1')
mapping = [c - 1 for c in args.channels]  # Channel numbers start with 1
q = queue.Queue()


if args.samplerate is None:
    device_info = sd.query_devices(args.device, 'input')
    args.samplerate = device_info['default_samplerate']

audiobuf = np.zeros((0, len(args.channels)))

hpcp = np.zeros((1, 12))
corr = np.zeros((1, PHRASEBUF_LEN))
crp = estd.ChromaCrossSimilarity(frameStackSize=9,
                                 frameStackStride=1,
                                 binarizePercentile=0.095,
                                 oti=True)

tracker = MotifTracker()
mcColors = defaultdict(lambda: random.choice(list(mcolors.CSS4_COLORS.keys())))

fig, axs = plt.subplots(2, 1, figsize=(20, 8))
hpcp_subplot = axs[0]
corr_subplot = axs[1]


def audio_callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    q.put(indata[::args.downsample, mapping])


stream = sd.InputStream(
    device=args.device, channels=max(args.channels),
    samplerate=args.samplerate, callback=audio_callback)

hpcp_im = hpcp_subplot.imshow(hpcp.T, aspect='auto', interpolation='none',
                              origin='lower', animated=False,
                              vmin=0., vmax=1.)
hpcp_secax = hpcp_subplot.secondary_xaxis('top',
                                          functions=(
                                              lambda f: f / FRAMES_PER_SECOND,
                                              lambda s: s * FRAMES_PER_SECOND))
hpcp_secax.set_xlabel('Seconds')

corr_im = corr_subplot.imshow(corr, aspect='auto', interpolation='none',
                              origin='lower', animated=False,
                              vmin=0, vmax=2)
corr_secax = corr_subplot.secondary_xaxis('top',
                                          functions=(
                                              lambda f: f / FRAMES_PER_SECOND,
                                              lambda s: s * FRAMES_PER_SECOND))
corr_secax.set_xlabel('Seconds')
lines = []


def update_plots(frame):
    """This is called by matplotlib for each plot update.

    Typically, audio callbacks happen more frequently than plot updates,
    therefore the queue tends to contain multiple blocks of audio data.

    """
    global audiobuf
    global hpcp
    global lines
    global corr
    while True:
        try:
            data = q.get_nowait()
        except queue.Empty:
            break
        audiobuf = np.append(audiobuf, data)

    if len(audiobuf) > AUDIOBUF_WINDOW_LEN:
        window = audiobuf[:AUDIOBUF_WINDOW_LEN]
        audiobuf = audiobuf[AUDIOBUF_WINDOW_LEN + 1:]

        hpcp_window = hpcpgram(
            window.ravel(), sampleRate=args.samplerate, hopSize=HOPSIZE)
        hpcp = np.concatenate((hpcp, hpcp_window), axis=0)

        if hpcp.shape[0] > PHRASEBUF_LEN * 2 + 1:
            pair_crp = crp(hpcp[:-PHRASEBUF_LEN], hpcp[-PHRASEBUF_LEN + 1:])

            corr = correlate(pair_crp, kernel, mode='same')
            # print(corr.max())
            corr = np.where(corr < SIMILARITY_THRESHOLD, 0, corr)
            # corr = cv2.GaussianBlur(corr, (63, 63), 0)
            corr = skeletonize(corr)
            corr = np.uint8(corr)
            lines = cv2.HoughLinesP(corr, rho=1, theta=np.pi/180,
                                    threshold=100, minLineLength=50, maxLineGap=30)

        hpcp_im.set_data(hpcp.T)
        hpcp_im.set_extent([0, hpcp.shape[0], 0, 12])

        corr_im.set_data(corr.T)
        corr_im.set_extent([0, hpcp.shape[0], 0, corr.shape[1]])

        if lines is not None:
            print("lines")
            for l in lines:
                print(l)
            for pair in lines:
                y1, x1, y2, x2 = pair
                # print(x1, x2, y1, y2)
                # hpcp_subplot.plot([x1, x2], [0, 1], color='red')
                # hpcp_subplot.plot([hpcp.shape[0] - PHRASEBUF_LEN + y1,
                #                   hpcp.shape[0] - PHRASEBUF_LEN + y2], [0, 1], color='blue')
                # tracker.noticeSimilarities(
                #     (x1, x2), (hpcp.shape[0] - PHRASEBUF_LEN + y1, hpcp.shape[0] - PHRASEBUF_LEN + y2))
                tracker.noticeSimilarities(
                    (x1, x2), (hpcp.shape[0] - PHRASEBUF_LEN + y1, hpcp.shape[0] - PHRASEBUF_LEN + y2))
            print('---')
            print('motifs')
            # motifClasses = set([m.motifClass for m in tracker.motif.items()])
            # mcColors = dict()
            # for mc in motifClasses:
            #     mcColors[mc] = random.choice(list(mcolors.CSS4_COLORS.keys()))

            for line in list(corr_subplot.lines):
                line.remove()
            for motif in tracker.motifs.values():
                corr_subplot.plot(
                    [motif.x1, motif.x2], [10, 30], color=mcColors[motif.motifClass], linewidth=20)

            corr_im.set_extent([0, hpcp.shape[0], 0, corr.shape[1]])
            for m in dict(sorted(tracker.motifs.items())).values():
                print(f"{m.x1} {m.x2} ({m.motifClass})\n")
            print("---")

    return [hpcp_im]


ani = FuncAnimation(
    fig, update_plots, interval=args.interval, blit=False)
with stream:
    plt.show()
