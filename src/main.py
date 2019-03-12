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

THROTTLE_BLOCK_HEIGHT = 1/20
THROTTLE_BLOCK_GAP = 1/60

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
    add_throttle_bar(frame, SENSOR_DATA.throttle(t))
    #add_throttle_bar(frame, -0.3)
    #plt.figure(1)
    #plt.imshow(frame)
    #plt.show()
    #sys.exit(0)
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


def add_throttle_bar(frame, throttle):
    frame_height, frame_width, _ = frame.shape

    bar_height = int(frame_height*9/10)
    bar_width = int(frame_width/20)

    bar_y_position = int(frame_height*1/20)
    bar_x_position = int(frame_width*1/30)

    bar_center = int(bar_height*2/3)

    shape_im = PIL.Image.new('RGB', (bar_width, bar_height))
    draw = ImageDraw.Draw(shape_im)

    top_gradient = np.tile(np.linspace(1, 0, bar_center), (bar_width, 1)).T
    bottom_gradient = np.tile(np.linspace(0, 1, bar_height - bar_center), (bar_width, 1)).T

    draw.polygon([
        (0, 0),
        (bar_width, 0),
        (bar_width/2, bar_center),
        (bar_width, bar_height),
        (0, bar_height)
    ], fill=(1, 1, 1))
    mask = PIL_to_npimage(shape_im)

    bar = np.empty((bar_height, bar_width, 3), dtype='uint8')

    bar[0:bar_center, :, 0] = top_gradient*255
    bar[0:bar_center, :, 1] = (1 - top_gradient)*255
    bar[0:bar_center, :, 2] = 0

    bar[bar_center:bar_height, :, 0] = bottom_gradient*255
    bar[bar_center:bar_height, :, 1] = (1 - bottom_gradient)*255
    bar[bar_center:bar_height, :, 2] = 0

    throttle_bar_mask = get_throttle_bar_mask(throttle, bar.shape, bar_center, bar_height)

    bar *= mask
    bar *= throttle_bar_mask

    overlay_image(frame, bar, bar_y_position, bar_x_position)


def get_throttle_bar_mask(throttle, shape, bar_center, bar_height):
    mask = np.zeros(shape, dtype='uint8')
    block_height = int(THROTTLE_BLOCK_HEIGHT*bar_center)
    block_gap = int(THROTTLE_BLOCK_GAP*bar_center)
    if throttle >= 0:
        for i in range(int(throttle*bar_center/block_height)):
            bottom_index = bar_center - i*block_height
            top_index = bar_center - (i + 1)*block_height + block_gap
            mask[top_index:bottom_index, :, :] = 1
    else:
        for i in range(int(-throttle*(bar_height - bar_center)/block_height)):
            bottom_index = bar_center + i*block_height
            top_index = bar_center + (i + 1)*block_height - block_gap
            mask[bottom_index:top_index, :, :] = 1

    width = shape[1]

    mask[block_gap:shape[0]-block_gap, 0:int(width/8), :] = 1

    return mask


def overlay_image(frame, image, y, x):
    np_image = PIL_to_npimage(image)
    height, width, _ = np_image.shape
    mask = np.sum(np_image, axis=-1)
    new_shit = np.empty(np_image.shape)
    new_shit[:, :, 0] = mask
    new_shit[:, :, 1] = mask
    new_shit[:, :, 2] = mask
    maximum = np.max(new_shit)
    if maximum == 0:
        return
    new_shit /= maximum
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

