mitsuba mts_shadow.xml -Dw=512 -Dh=512 -Dsamples=128 -Dori="0,1.04671,1.93185" -Dtarget="0,-0.37529,-0.926908" -Dup="0,0.859158,-0.347859" -Dground="/home/ysheng/Dataset/models/ground/ground.obj" -Dmodel="/home/ysheng/Dataset/models/notsimulated_combine_male_short_outfits_genesis8_armani_casualoutfit03_Base_Pose_Standing_A/notsimulated_combine_male_short_outfits_genesis8_armani_casualoutfit03_Base_Pose_Standing_A.obj" -Dibl="../test_pattern.png" -Dworld="0.0048338 0 0 0 0 0.0048338 0 0 0 0 0.0048338 0 0.0068885 0.0110922 -0.0205234 1"
mitsuba mts_final.xml -Dw=512 -Dh=512 -Dsamples=128 -Dori="0,1.04671,1.93185" -Dtarget="0,-0.37529,-0.926908" -Dup="0,0.859158,-0.347859" -Dground="/home/ysheng/Dataset/models/ground/ground.obj" -Dmodel="/home/ysheng/Dataset/models/notsimulated_combine_male_short_outfits_genesis8_armani_casualoutfit03_Base_Pose_Standing_A/notsimulated_combine_male_short_outfits_genesis8_armani_casualoutfit03_Base_Pose_Standing_A.obj" -Dibl="../test_pattern.png" -Dworld="0.0048338 0 0 0 0 0.0048338 0 0 0 0 0.0048338 0 0.0068885 0.0110922 -0.0205234 1"
mtsutil tonemap -m=10.0 mts_final.exr
mtsutil tonempa mts_shadow.exr
