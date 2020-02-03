frame_rate=24
cd real_human
ffmpeg -y -r $frame_rate -i %07d.png $1
cd ..
cd training_human_own_light
ffmpeg -y -r $frame_rate -i %07d.png $1
cd ..
cd training_simple_obj
ffmpeg -y -r $frame_rate -i %07d.png $1




