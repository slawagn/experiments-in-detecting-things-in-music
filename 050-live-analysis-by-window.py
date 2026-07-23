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
# fmt: on


AUDIOBUF_LEN = 12000
PHRASEBUF_LEN = 128
PHRASE_STEP_SIZE = int(PHRASEBUF_LEN / 2)


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
pair_crp = np.zeros((1, PHRASEBUF_LEN))
crp = estd.ChromaCrossSimilarity(frameStackSize=9,
                                 frameStackStride=1,
                                 binarizePercentile=0.095,
                                 oti=True)
similarity_distances = np.empty((1, 1))

fig, axs = plt.subplots(2, 1, figsize=(32, 16))
hpcp_subplot = axs[0]
similarities_subplot = axs[1]


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
similarities_line, = similarities_subplot.plot(similarity_distances)


def update_plots(frame):
    """This is called by matplotlib for each plot update.

    Typically, audio callbacks happen more frequently than plot updates,
    therefore the queue tends to contain multiple blocks of audio data.

    """
    global audiobuf
    global hpcp
    global similarity_distances
    while True:
        try:
            data = q.get_nowait()
        except queue.Empty:
            break
        audiobuf = np.append(audiobuf, data)

    if len(audiobuf) > AUDIOBUF_LEN:
        window = audiobuf[:AUDIOBUF_LEN]
        audiobuf = audiobuf[AUDIOBUF_LEN + 1:]

        hpcp_window = hpcpgram(window.ravel(), sampleRate=args.samplerate)
        hpcp = np.concatenate((hpcp, hpcp_window), axis=0)

        if hpcp.shape[0] > PHRASEBUF_LEN * 2 + 1:

            past_hpcp = hpcp[:-PHRASEBUF_LEN]
            current_hpcp = hpcp[-PHRASEBUF_LEN + 1:]

            similarity_distances = np.empty((2, 0))
            for i in range(0, len(past_hpcp) - PHRASEBUF_LEN, PHRASE_STEP_SIZE):
                current_pair_crp = crp(
                    current_hpcp, past_hpcp[i:i + PHRASEBUF_LEN])
                current_score_matrix, current_pair_distance = estd.CoverSongSimilarity(disOnset=0.5,
                                                                                       disExtension=0.5,
                                                                                       alignmentType='serra09',
                                                                                       distanceType='asymmetric')(current_pair_crp)
                similarity_distances = np.concat(
                    (similarity_distances, np.array([[i], [current_pair_distance]])), axis=1)

            similarities_line.set_data(similarity_distances)

            # similarities_subplot.set_yscale('symlog')
            similarities_subplot.set_xlim(0, past_hpcp.shape[0])
            similarities_subplot.set_ylim(0, 2)

    # return [hpcp_im, similarities_line]


ani = FuncAnimation(
    fig, update_plots, interval=args.interval, blit=False)
with stream:
    plt.show()
