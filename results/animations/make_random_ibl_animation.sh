#!/bin/bash
render_animations() {
for j in {3..12..1}
  do
      prefix="$1$j"
      echo "$prefix"_%07d
      echo "$1_$j_%07d"
      ffmpeg -y -r $frame_rate -i "$prefix"_%07d.png -c:v libx264 -pix_fmt yuv420p "$prefix".mp4
  done
}

frame_rate=$2
cd experiments
render_animations $1
cd ../training_human_own_light
render_animations $1
cd ../training_simple_obj
render_animations $1
