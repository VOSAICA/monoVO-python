import numpy as np
import cv2

from visual_odometry import PinholeCamera, VisualOdometry
from scipy.spatial.transform import Rotation as R

cam = PinholeCamera(1226.0, 370.0, 718.8560, 718.8560, 607.1928, 185.2157)
vo = VisualOdometry(
    cam, '/home/raoshi/Documents/datasets/KITTI_odometry_poses/06.txt')

traj = np.zeros((1200, 1200, 3), dtype=np.uint8)

for img_id in range(10000):
    try:
        img = cv2.imread(
            '/home/raoshi/Documents/datasets/KITTI_odometry_color/06/image_2/'
            + str(img_id).zfill(6) + '.png', cv2.IMREAD_GRAYSCALE)
        vo.update(img, img_id)

        cur_t = vo.cur_t
        cur_r = vo.cur_R
        r = R.from_matrix(cur_r)
        r = r.as_quat()
        qx, qy, qz, qw = r[0], r[1], r[2], r[3]
        i = img_id
        if (img_id > 2):
            x, y, z = cur_t[0], cur_t[1], cur_t[2]
        else:
            x, y, z = 0., 0., 0.
        draw_x, draw_y = int(x) + 400, int(z) + 500
        with open("results.txt", 'a') as outposeF:
            outposeF.write('%f %f %f %f %f %f %f %f\n' %
                           (i, x, y, z, qx, qy, qz, qw))
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
            cv2.imwrite('map.png', traj)
            exit()
