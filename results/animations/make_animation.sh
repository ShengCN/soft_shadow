frame_rate=$2
cd experiments
ffmpeg -y -r $frame_rate -i %07d.png -c:v libx264 -pix_fmt yuv420p $1
cd ..
cd real_human
ffmpeg -y -r $frame_rate -i %07d.png -c:v libx264 -pix_fmt yuv420p $1
cd ..
cd training_human_own_light
ffmpeg -y -r $frame_rate -i %07d.png -c:v libx264 -pix_fmt yuv420p $1
cd ..
cd training_simple_obj
ffmpeg -y -r $frame_rate -i %07d.png -c:v libx264 -pix_fmt yuv420p $1




