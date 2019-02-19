#!/usr/bin/env python3

import scipy.interpolate as interpolate
import sys
from moviepy.video.io.bindings import PIL_to_npimage
import matplotlib.pyplot as plt
import moviepy.editor as edit
import numpy as np
import argparse
import PIL
from PIL import ImageFont, ImageDraw
import pdb

FONT_SIZE = 200

MAX_SPEED = 120.0
global FONT
FONT = None

class SensorData:
    speed = None
    throttle = None
    steering = None

SENSOR_DATA = SensorData()


def add_telemetry_data(get_frame, t):
    frame = get_frame(t)
    height, _, _, = frame.shape
    global FONT
    if FONT is None:
        FONT = ImageFont.FreeTypeFont("../fonts/PEPSI_pl.ttf", int(FONT_SIZE*(height/1440)))

    add_speed_text(frame, SENSOR_DATA.speed(t))
    plt.figure(1)
    plt.imshow(frame)
    plt.show()
    sys.exit(0)
    return frame


def add_speed_text(frame, speed):
    im = PIL.Image.new('RGB', (int(FONT_SIZE*2.7), int(FONT_SIZE*1.5)))
    draw = ImageDraw.Draw(im)
    red_channel = min(int((speed / MAX_SPEED) * 255), 255)
    draw.text((2, 0), "{}\nkm/h".format(int(speed)),
              (red_channel, 255 - red_channel, 0),
              font=FONT)

    height, width, _ = frame.shape

    overlay_image(frame, im, int(height*(31/40)), int(width*(16/20)))


def overlay_image(frame, image, y, x):
    np_image = PIL_to_npimage(image)
    height, width, _ = np_image.shape
    mask = np.sum(np_image, axis=-1)
    new_shit = np.empty(np_image.shape)
    new_shit[:, :, 0] = mask
    new_shit[:, :, 1] = mask
    new_shit[:, :, 2] = mask
    new_shit /= np.max(new_shit)
    patch = frame[y:y+height, x:x+width, :]
    patch[:, :, :] = (1 - new_shit)*patch + new_shit*np_image


def init_sensor_data(data_file_name):
    data = None
    with open(data_file_name) as f:
        data = [tuple(float(c) for c in x.split(',')) for x in f.readlines()]

    data_array = np.array(data)
    SENSOR_DATA.speed = interpolate.interp1d(data_array[:, 0], data_array[:, 1])
    SENSOR_DATA.throttle = interpolate.interp1d(data_array[:, 0], data_array[:, 2])
    SENSOR_DATA.steering = interpolate.interp1d(data_array[:, 0], data_array[:, 3])


def main():
    parser = argparse.ArgumentParser(description='Create video with telemetry overlay.')
    parser.add_argument('-m', '--media', required=True, type=str,
                       help='Path to the video file')
    parser.add_argument('-d', '--data', required=True, type=str,
                       help='Path to the telemetry data')

    args = parser.parse_args()

    init_sensor_data(args.data)

    

    videoclip = edit.VideoFileClip(args.media)
    videoclip = videoclip.fl(add_telemetry_data)
    videoclip.write_videofile('output.mp4', threads=4)


if __name__ == '__main__':
    main()

