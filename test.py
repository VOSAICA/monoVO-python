import numpy as np
import cv2

from visual_odometry import PinholeCamera, VisualOdometry
from scipy.spatial.transform import Rotation as R

num = str("08")

imgTemp = cv2.imread(
            '/home/raoshi/Documents/datasets/KITTI_odometry_color/' + num + '/image_2/'
            + str(1).zfill(6) + '.png', cv2.IMREAD_GRAYSCALE)
h = imgTemp.shape[0]
w = imgTemp.shape[1]

cam = PinholeCamera(float(w), float(h), 718.8560, 718.8560, 607.1928, 185.2157)
vo = VisualOdometry(
    cam, '/home/raoshi/Documents/datasets/KITTI_odometry_poses/' + num +'.txt')

traj = np.zeros((1200, 1200, 3), dtype=np.uint8)

for img_id in range(10000):
    try:
        img = cv2.imread(
            '/home/raoshi/Documents/datasets/KITTI_odometry_color/' + num +'/image_2/'
            + str(img_id).zfill(6) + '.png', cv2.IMREAD_GRAYSCALE)
        vo.update(img, img_id)

        cur_t = vo.cur_t
        cur_r = vo.cur_R
        r = R.from_matrix(cur_r)
        r = r.as_quat()
        qx, qy, qz, qw = r[0], r[1], r[2], r[3]
        r = cur_r
        r1, r2, r3, r4, r5, r6, r7, r8, r9 = r[0][0], r[0][1], r[0][2], r[1][0], r[1][1], r[1][2], r[2][0], r[2][1], r[2][2]
        i = img_id
        if (img_id > 2):
            x, y, z = cur_t[0], cur_t[1], cur_t[2]
        else:
            x, y, z = 0., 0., 0.
        draw_x, draw_y = int(x) + 400, int(z) + 500
        with open("results.txt", 'a') as outposeF:
            outposeF.write('%f %f %f %f %f %f %f %f\n' %
                           (i, x, y, z, qx, qy, qz, qw))
        with open("results1.txt", 'a') as outposeF:
            outposeF.write('%f %f %f %f %f %f %f %f %f %f %f %f\n' %
                           (r1, r2, r3, x, r4, r5, r6, y, r7, r8, r9, z))
        true_x, true_y = int(vo.trueX) + 400, int(vo.trueZ) + 500

        cv2.circle(traj, (draw_x, draw_y), 1,
                   (img_id * 255 / 4540, 255 - img_id * 255 / 4540, 0), 1)
        cv2.circle(traj, (true_x, true_y), 1, (0, 0, 255), 2)
        cv2.rectangle(traj, (10, 20), (600, 60), (0, 0, 0), -1)
        text = "Coordinates: x=%2fm y=%2fm z=%2fm" % (x, y, z)
        cv2.putText(traj, text, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1,
                    (255, 255, 255), 1, 8)

        cv2.imshow('Road facing camera', img)
        cv2.imshow('Trajectory', traj)
        cv2.waitKey(1)

    except:
        if img_id > 10:
            ("ended")
            cv2.imwrite('map.png', traj)
            exit()
