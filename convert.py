from scipy.spatial.transform import Rotation as R

groundtruth = open(
    '/home/raoshi/Documents/datasets/KITTI_odometry_poses/08.txt', 'r')
converted = open("groundtruth.txt", 'w')
temp = groundtruth.readline()
i = 0
while True:
    i += 1
    temp = groundtruth.readline()
    temp = temp.split()
    R11 = float(temp[0])
    R12 = float(temp[1])
    R13 = float(temp[2])
    x = float(temp[3])
    R21 = float(temp[4])
    R22 = float(temp[5])
    R23 = float(temp[6])
    y = float(temp[7])
    R31 = float(temp[8])
    R32 = float(temp[9])
    R33 = float(temp[10])
    z = float(temp[11])
    r = R.from_matrix([[R11, R12, R13], [R21, R22, R23], [R31, R32, R33]])
    r = r.as_quat()
    qx, qy, qz, qw = r[0], r[1], r[2], r[3]
    converted.write('%f %f %f %f %f %f %f %f\n' % (i, x, y, z, qx, qy, qz, qw))
