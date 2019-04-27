#!/bin/bash
ffprobe -v quiet -print_format json -show_streams -show_entries stream=index,codec_type:stream_tags=creation_time:format_tags=creation_time $1
