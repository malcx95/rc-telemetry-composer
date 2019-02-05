#!/usr/bin/env python3
import moviepy.editor as edit
import argparse


def add_telemetry_data(get_frame, t):
    frame = get_frame(t)
    frame[:, 500:1000, :] = (0, 255, 0)
    return frame


def main():
    parser = argparse.ArgumentParser(description='Create video with telemetry overlay.')
    parser.add_argument('-m', '--media', required=True, type=str,
                       help='Path to the video file')
    parser.add_argument('-d', '--data', required=True, type=str,
                       help='Path to the telemetry data')

    args = parser.parse_args()

    videoclip = edit.VideoFileClip(args.media)
    videoclip = videoclip.fl(add_telemetry_data)
    videoclip.write_videofile('output.mp4', threads=4)


if __name__ == '__main__':
    main()

