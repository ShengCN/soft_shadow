cd real_human
ffmpeg -i %07d.png $1
cd ..
cd training_human_own_light
ffmpeg -i %07d.png $1
cd ..
cd training_simple_obj
ffmpeg -i %07d.png $1




