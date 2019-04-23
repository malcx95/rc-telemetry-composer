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
import subprocess
import pdb
import tempfile
import os
from datetime import datetime
import json


FONT_SIZE = 200
SMALL_FONT_SIZE = 40

MAX_SPEED = 120.0
global FONT
FONT = None
global SMALL_FONT
SMALL_FONT = None
global TEXT_FRAME
TEXT_FRAME = None

global THROTTLE_BAR
THROTTLE_BAR = None
global STEERING_BAR
STEERING_BAR = None

THROTTLE_BLOCK_HEIGHT = 1/20
THROTTLE_BLOCK_GAP = 1/60

THROTTLE_BAR_HEIGHT = 9/10
THROTTLE_BAR_WIDTH = 1/20
THROTTLE_BAR_CENTER = 2/3

STEERING_BAR_WIDTH = 4/20
STEERING_BAR_HEIGHT = 1/20


class SensorData:
    speed = None
    throttle = None
    steering = None
    time_offset = None


SENSOR_DATA = SensorData()


def add_telemetry_data(get_frame, t):
    offset = SENSOR_DATA.time_offset
    frame = get_frame(t)
    height, _, _, = frame.shape
    global FONT
    global SMALL_FONT
    global TEXT_FRAME
    global THROTTLE_BAR
    global STEERING_BAR
    if FONT is None:
        FONT = ImageFont.FreeTypeFont("../fonts/PEPSI_pl.ttf", int(FONT_SIZE*(height/1440)))
        SMALL_FONT = ImageFont.FreeTypeFont("../fonts/PEPSI_pl.ttf", int(SMALL_FONT_SIZE*(height/1440)))
        TEXT_FRAME = get_static_text(frame.shape)
        THROTTLE_BAR = prepare_throttle_bar(frame)
        STEERING_BAR = prepare_steering_bar(frame)

    add_speed_text(frame, SENSOR_DATA.speed(t + offset))
    add_throttle_bar(frame, SENSOR_DATA.throttle(t + offset), np.copy(THROTTLE_BAR))
    #add_steering_bar(frame, SENSOR_DATA.steering(t + offset), np.copy(STEERING_BAR))
    add_steering_bar(frame, -1.0, np.copy(STEERING_BAR))
    #add_throttle_bar(frame, 1.0)

    overlay_image(frame, TEXT_FRAME, 0, 0)
    plt.figure(1)
    plt.imshow(frame)
    plt.show()
    sys.exit(0)
    return frame


def get_static_text(shape):
    height, width, _ = shape
    im = PIL.Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(im)

    draw.text((width*35/1000, height*4/100), "throttle", (255, 0, 0), font=SMALL_FONT)
    draw.text((width*35/1000, height*94/100), "brake", (255, 0, 0), font=SMALL_FONT)

    return PIL_to_npimage(im)


def add_speed_text(frame, speed):
    height, width, _ = frame.shape

    im = PIL.Image.new('RGB', (int(FONT_SIZE*2.7*(height/1440)),
                               int(FONT_SIZE*1.5*(height/1440))))
    draw = ImageDraw.Draw(im)
    red_channel = min(int((speed / MAX_SPEED) * 255), 255)
    draw.text((2, 0), "{}\nkm/h".format(int(speed)),
              (red_channel, 255 - red_channel, 0),
              font=FONT)

    overlay_image(frame, im, int(height*(31/40)), int(width*(15/20)))


def prepare_throttle_bar(frame):
    frame_height, frame_width, _ = frame.shape

    bar_height = int(frame_height*THROTTLE_BAR_HEIGHT)
    bar_width = int(frame_width*THROTTLE_BAR_WIDTH)
    bar_center = int(bar_height*THROTTLE_BAR_CENTER)

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

    bar *= mask

    return bar


def add_throttle_bar(frame, throttle, bar):
    frame_height, frame_width, _ = frame.shape

    bar_height, bar_width, _ = bar.shape

    bar_center = int(bar_height*THROTTLE_BAR_CENTER)

    bar_y_position = int(frame_height*1/20)
    bar_x_position = int(frame_width*1/30)

    throttle_bar_mask = get_throttle_bar_mask(throttle, bar.shape, bar_center, bar_height)

    bar *= throttle_bar_mask

    overlay_image(frame, bar, bar_y_position, bar_x_position)


def prepare_steering_bar(frame):
    frame_height, frame_width, _ = frame.shape

    bar_width = int(frame_width*STEERING_BAR_WIDTH)*2
    bar_height = int(frame_height*STEERING_BAR_HEIGHT)
    bar_center = bar_width//2

    bar = np.zeros((bar_height, bar_width, 3), dtype='uint8')

    left_gradient = np.tile(np.linspace(1, 0, bar_center), (bar_height, 1))
    right_gradient = np.tile(np.linspace(0, 1, bar_center), (bar_height, 1))

    bar[:, :bar_center, 0] = left_gradient*255
    bar[:, :bar_center, 1] = 0
    bar[:, :bar_center, 2] = (1 - left_gradient)*255

    bar[:, bar_center:, 0] = right_gradient*255
    bar[:, bar_center:, 1] = 0
    bar[:, bar_center:, 2] = (1 - right_gradient)*255

    shape_im = PIL.Image.new('RGB', (bar_width, bar_height))
    draw = ImageDraw.Draw(shape_im)

    return bar


def add_steering_bar(frame, steering, bar):
    frame_height, frame_width, _ = frame.shape

    bar_height, bar_width, _ = bar.shape
    bar_center = bar_width//2

    bar_x_position = frame_width//2 - bar_width//2
    bar_y_position = int(frame_height*221/240)

    mask = get_steering_bar_mask(steering, bar.shape, bar_center)
    bar *= mask

    overlay_image(frame, bar, bar_y_position, bar_x_position)


def get_steering_bar_mask(steering, shape, bar_center):
    mask = np.zeros(shape, dtype='uint8')
    height = shape[0]
    mask[int(height*5/6):, :, :] = 1
    if steering < 0:
        mask[:, bar_center:bar_center + int(bar_center*-steering), :] = 1
    else:
        mask[:, bar_center - int(bar_center*steering):bar_center, :] = 1
    return mask


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

    mask[block_gap*2:shape[0]-block_gap*2, 0:int(width/8), :] = 1

    return mask


def overlay_image(frame, image, y, x):
    np_image = PIL_to_npimage(image)
    height, width, _ = np_image.shape
    mask = np.sum(np_image, axis=-1)
    new_img = np.empty(np_image.shape)
    new_img[:, :, 0] = mask
    new_img[:, :, 1] = mask
    new_img[:, :, 2] = mask
    maximum = np.max(new_img)
    if maximum == 0:
        return
    new_img /= maximum
    patch = frame[y:y+height, x:x+width, :]
    patch[:, :, :] = (1 - new_img)*patch + new_img*np_image


def rpm_to_speed(rpm, ten_count_distance):
    return 3.6*(rpm/60)*ten_count_distance


def time_to_seconds(time):
    time_split = time[1:].split(':')
    hour = int(time_split[0])
    minute = int(time_split[1])
    second = float(time_split[2])
    return hour*3600 + minute*60 + second


def is_same_day(date1, date2):
    return date1.year == date2.year and \
            date1.month == date2.month and \
            date1.day == date2.day


def init_sensor_data(data_path, video_date, ten_count_distance):
    csv_files = []
    for root, _, files in os.walk(data_path):
        csv_files.extend([(root, f) for f in files if f.endswith('.csv')])

    data = []
    first_date = None
    for root, csv_file in sorted(csv_files):
        date = datetime.strptime("20" + csv_file, "%Y%m%d%H%M%S.csv")
        if is_same_day(date, video_date):
            if first_date is None:
                first_date = date
            lines = None
            with open(os.path.join(root, csv_file)) as f:
                lines = f.readlines()
                #data = [tuple(float(c) for c in x.split(',')) for x in f.readlines()[6:]]

            for line in lines[5:]:
                line_split = line.split(',')
                time = time_to_seconds(line_split[2])
                steering = float(line_split[3])/100
                throttle = float(line_split[4])/100
                rpm = float(line_split[5])
                speed = rpm_to_speed(int(rpm), ten_count_distance)

                data.append((time, steering, throttle, speed))

    data_array = np.array(data)
    SENSOR_DATA.speed = interpolate.interp1d(data_array[:, 0], data_array[:, 3],
                                             fill_value=0, bounds_error=False)
    SENSOR_DATA.throttle = interpolate.interp1d(data_array[:, 0], data_array[:, 2],
                                                fill_value=0, bounds_error=False)
    SENSOR_DATA.steering = interpolate.interp1d(data_array[:, 0], data_array[:, 1],
                                                fill_value=0, bounds_error=False)
    SENSOR_DATA.time_offset = video_date - first_date
    #pdb.set_trace()


def get_mp4_creation_date(mediafile):
    output = subprocess.check_output(
            ["./mp4info.sh", mediafile],
            stderr=subprocess.STDOUT, timeout=3,
            universal_newlines=True)
    metadata = json.loads(output)
    return datetime.strptime(metadata["streams"][0]["tags"]["creation_time"],
                             "%Y-%m-%d %H:%M:%S")


def main():
    parser = argparse.ArgumentParser(description='Create video with telemetry overlay.')
    parser.add_argument('-m', '--media', required=True, type=str,
                       help='Path to the video file')
    parser.add_argument('-d', '--data', required=True, type=str,
                       help='Path to the telemetry data')
    parser.add_argument('-t', '--ten-count-distance', type=float,
                        help='Number of meters the vehicle travels every 10 revolutions',
                        default=0.187)
    parser.add_argument('-o', '--output', help='The output file to create',
                        default='output.mp4')
    parser.add_argument('-r', '--downsample', type=int, help='Factor to downsample',
                        default=1)
    parser.add_argument('-s', '--offset', type=float, help='Number of seconds to advance the telemetry data', default=0.0)

    args = parser.parse_args()

    video_date = get_mp4_creation_date(args.media)

    init_sensor_data(args.data, video_date, args.ten_count_distance)
    SENSOR_DATA.time_offset = args.offset

    width, height = edit.VideoFileClip(args.media).size
    videoclip = edit.VideoFileClip(args.media, target_resolution=(height//args.downsample,
                                                                  width//args.downsample))

    videoclip = videoclip.fl(add_telemetry_data)
    videoclip.write_videofile(args.output, threads=6)


if __name__ == '__main__':
    main()

