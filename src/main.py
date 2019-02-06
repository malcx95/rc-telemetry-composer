#!/usr/bin/env python3

import scipy.interpolate as interpolate
import moviepy.editor as edit
import numpy as np
import argparse

class SensorData:
    speed = None
    throttle = None
    steering = None

SENSOR_DATA = SensorData()

def add_telemetry_data(get_frame, t):
    frame = get_frame(t)
    frame[0:int(SENSOR_DATA.speed(t)), :, :] = (0, 255, 0)
    return frame


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

