# fmt: off
import matplotlib
matplotlib.use('TkAgg')

import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
import queue
import sys

import essentia.standard as estd
from essentia.pytools.spectral import hpcpgram
from scipy.signal import correlate
import cv2
# fmt: on


AUDIOBUF_WINDOW_LEN = 48000
PHRASEBUF_LEN = 150

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
corr_im = corr_subplot.imshow(corr, aspect='auto', interpolation='none',
                              origin='lower', animated=False,
                              vmin=0, vmax=127)
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

        hpcp_window = hpcpgram(window.ravel(), sampleRate=args.samplerate)
        hpcp = np.concatenate((hpcp, hpcp_window), axis=0)

        if hpcp.shape[0] > PHRASEBUF_LEN * 2 + 1:
            pair_crp = crp(hpcp[:-PHRASEBUF_LEN], hpcp[-PHRASEBUF_LEN + 1:])

            corr = np.uint8(correlate(pair_crp, kernel))
            print(corr.max())
            corr = np.where(corr < 50, 0, corr)
            corr = cv2.GaussianBlur(corr, (63, 63), 0)
            lines = cv2.HoughLinesP(corr, rho=1, theta=np.pi/180,
                                    threshold=100, minLineLength=50, maxLineGap=10)

        hpcp_im.set_data(hpcp.T)
        hpcp_im.set_extent([0, hpcp.shape[0], 0, 12])

        corr_im.set_data(corr.T)
        corr_im.set_extent([0, corr.shape[0], 0, corr.shape[1]])

        if lines is not None:
            for pair in lines:
                x1, y1, x2, y2 = pair
                corr_subplot.plot([y1, y2], [10, 20], color='red')
                corr_subplot.plot([hpcp.shape[0] - PHRASEBUF_LEN + x1,
                                  hpcp.shape[0] - PHRASEBUF_LEN + x2], [10, 20], color='blue')

    return [hpcp_im]


ani = FuncAnimation(
    fig, update_plots, interval=args.interval, blit=False)
with stream:
    plt.show()
