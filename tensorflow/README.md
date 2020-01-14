# C073_Head_cutting_from_point_cloud

# Results
Results will vary slightly for each time test.py is run. This is caused by the random down sampling of point clouds to 1024 points. 

# lay
test GT: pos error mean 0.007862 std 0.003721, ang error mean 4.484635 std 2.736516
val GT: pos error mean 0.013454 std 0.029660, ang error mean 6.432392 std 4.884108
train GT: pos error mean 0.007144 std 0.003334, ang error mean 3.422024 std 1.917789

# hang
train GT: pos error mean 0.025914 std 0.047910, ang error mean 3.250849 std 1.887006
val GT: pos error mean 0.033432 std 0.022227, ang error mean 13.981595 std 18.999093
test GT: pos error mean 0.034243 std 0.018757, ang error mean 12.051596 std 11.205911

# head
test GT: pos error mean 0.011675 std 0.006095, ang error mean 3.241150 std 1.774784
val GT: pos error mean 0.013109 std 0.006508, ang error mean 4.566085 std 2.460330
train GT: pos error mean 0.002753 std 0.001766, ang error mean 0.362097 std 0.228558

# Differences from implementation used for paper
1. More data augmentation (minimal effect)
2. Lacks variations used to produce secondary results