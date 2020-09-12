import numpy as np
import cv2
import torch
from models.detecting import SuperpointDetector
from models.matching import Matching as SuperglueMatcher

STAGE_FIRST_FRAME = 0
STAGE_SECOND_FRAME = 1
STAGE_DEFAULT_FRAME = 2
kMinNumFeature = 1500

lk_params = dict(
    winSize=(21, 21),
    # maxLevel = 3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


def frame2tensor(frame, device):
    return torch.from_numpy(frame / 255.).float()[None, None].to(device)


class PinholeCamera:
    def __init__(self,
                 width,
                 height,
                 fx,
                 fy,
                 cx,
                 cy,
                 k1=0.0,
                 k2=0.0,
                 p1=0.0,
                 p2=0.0,
                 k3=0.0):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.distortion = (abs(k1) > 0.0000001)
        self.d = [k1, k2, p1, p2, k3]


class VisualOdometry:
    def __init__(self, cam, annotations):
        self.frame_stage = 0
        self.cam = cam
        self.new_frame = None
        self.last_frame = None
        self.cur_R = None
        self.cur_t = None
        self.px_ref = None
        self.px_cur = None
        self.focal = cam.fx
        self.pp = (cam.cx, cam.cy)
        self.trueX, self.trueY, self.trueZ = 0, 0, 0
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        # self.detector = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)

        self.detector = SuperpointDetector({
            'descriptor_dim': 256,
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': -1,
            'remove_borders': 4
        }).eval().to(self.device)
        self.tracker = SuperglueMatcher({
            'descriptor_dim': 256,
            'weights': 'indoor',
            'keypoint_encoder': [32, 64, 128, 256],
            'GNN_layers': ['self', 'cross'] * 9,
            'sinkhorn_iterations': 100,
            'match_threshold': 0.25
        }).eval().to(self.device)
        with open(annotations) as f:
            self.annotations = f.readlines()

    def featureTracking(self, image_ref, image_cur, px_ref):
        image_ref = frame2tensor(image_ref, self.device)
        image_cur = frame2tensor(image_cur, self.device)

        kp2, st, err = cv2.calcOpticalFlowPyrLK(
            image_ref, image_cur, px_ref, None,
            **lk_params)  # shape: [k,2] [k,1] [k,1]
        st = st.reshape(st.shape[0])
        kp1 = px_ref[st == 1]
        kp2 = kp2[st == 1]
        return kp1, kp2

    def getAbsoluteScale(self,
                         frame_id):  # specialized for KITTI odometry dataset
        ss = self.annotations[frame_id - 1].strip().split()
        x_prev = float(ss[3])
        y_prev = float(ss[7])
        z_prev = float(ss[11])
        ss = self.annotations[frame_id].strip().split()
        x = float(ss[3])
        y = float(ss[7])
        z = float(ss[11])
        self.trueX, self.trueY, self.trueZ = x, y, z
        return np.sqrt((x - x_prev) * (x - x_prev) + (y - y_prev) *
                       (y - y_prev) + (z - z_prev) * (z - z_prev))

    def processFirstFrame(self):
        data = {}
        data['image'] = frame2tensor(self.new_frame, self.device)
        data = {**data, **self.detector(data)}
        print(data)
        self.px_ref = data['keypoints'][0].cpu().detach().numpy()
        # self.px_ref = np.array([x.pt for x in self.px_ref], dtype=np.float32)
        self.frame_stage = STAGE_SECOND_FRAME

    def processSecondFrame(self):
        self.px_ref, self.px_cur = self.featureTracking(
            self.last_frame, self.new_frame, self.px_ref)
        E, mask = cv2.findEssentialMat(self.px_cur,
                                       self.px_ref,
                                       focal=self.focal,
                                       pp=self.pp,
                                       method=cv2.RANSAC,
                                       prob=0.999,
                                       threshold=1.0)
        _, self.cur_R, self.cur_t, mask = cv2.recoverPose(E,
                                                          self.px_cur,
                                                          self.px_ref,
                                                          focal=self.focal,
                                                          pp=self.pp)
        self.frame_stage = STAGE_DEFAULT_FRAME
        self.px_ref = self.px_cur

    def processFrame(self, frame_id):
        self.px_ref, self.px_cur = self.featureTracking(
            self.last_frame, self.new_frame, self.px_ref)
        E, mask = cv2.findEssentialMat(self.px_cur,
                                       self.px_ref,
                                       focal=self.focal,
                                       pp=self.pp,
                                       method=cv2.RANSAC,
                                       prob=0.999,
                                       threshold=1.0)
        _, R, t, mask = cv2.recoverPose(E,
                                        self.px_cur,
                                        self.px_ref,
                                        focal=self.focal,
                                        pp=self.pp)
        absolute_scale = self.getAbsoluteScale(frame_id)
        if (absolute_scale > 0.1):
            self.cur_t = self.cur_t + absolute_scale * self.cur_R.dot(t)
            self.cur_R = R.dot(self.cur_R)
        if (self.px_ref.shape[0] < kMinNumFeature):
            data = {}
            data['image'] = frame2tensor(self.new_frame, self.device)
            data = {**data, **self.detector(data)}
            self.px_cur = data['keypoints'][0].cpu().detach().numpy()
            # self.px_cur = np.array([x.pt for x in self.px_cur], dtype=np.float32)
        self.px_ref = self.px_cur

    def update(self, img, frame_id):
        assert (
            img.ndim == 2 and img.shape[0] == self.cam.height
            and img.shape[1] == self.cam.width
        ), "Frame: provided image has not the same size as the camera model or image is not grayscale"
        self.new_frame = img
        if (self.frame_stage == STAGE_DEFAULT_FRAME):
            self.processFrame(frame_id)
        elif (self.frame_stage == STAGE_SECOND_FRAME):
            self.processSecondFrame()
        elif (self.frame_stage == STAGE_FIRST_FRAME):
            self.processFirstFrame()
        self.last_frame = self.new_frame
