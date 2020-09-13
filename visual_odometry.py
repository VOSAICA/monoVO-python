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
        self.data = {}
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.detector = SuperpointDetector({
            'superpoint': {
                'descriptor_dim': 256,
                'nms_radius': 4,
                'keypoint_threshold': 0.005,
                'max_keypoints': -1,
                'remove_borders': 4
            }
        }).eval().to(self.device)
        self.tracker = SuperglueMatcher({
            'superglue': {
                'descriptor_dim': 256,
                'weights': 'indoor',
                'keypoint_encoder': [32, 64, 128, 256],
                'GNN_layers': ['self', 'cross'] * 9,
                'sinkhorn_iterations': 100,
                'match_threshold': 0.37
            }
        }).eval().to(self.device)

        with open(annotations) as f:
            self.annotations = f.readlines()

    @torch.no_grad()
    def featureTracking(self, image_ref, image_cur):
        image_cur = frame2tensor(image_cur, self.device)
        self.data['image1'] = image_cur
        tempData = {**self.data, **self.tracker(self.data)}
        kpts0 = tempData['keypoints0'][0].cpu().numpy()
        kpts1 = tempData['keypoints1'][0].cpu().numpy()
        matches = tempData['matches0'][0].cpu().numpy()
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        self.data.clear()
        self.data['image0'], self.data['keypoints0'] = tempData[
            'image1'], tempData['keypoints1']
        self.data['scores0'], self.data['descriptors0'] = tempData[
            'scores1'], tempData['descriptors1']
        return mkpts0, mkpts1

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

    @torch.no_grad()
    def processFirstFrame(self):
        self.data['image'] = frame2tensor(self.new_frame, self.device)
        self.data = {**self.data, **self.detector(self.data)}
        self.px_ref = self.data['keypoints'][0].cpu().detach().numpy()
        self.data = {**{k + '0': v for k, v in self.data.items()}}
        self.frame_stage = STAGE_SECOND_FRAME

    @torch.no_grad()
    def processSecondFrame(self):
        self.px_ref, self.px_cur = self.featureTracking(
            self.last_frame, self.new_frame)
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

    @torch.no_grad()
    def processFrame(self, frame_id):
        self.px_ref, self.px_cur = self.featureTracking(
            self.last_frame, self.new_frame)
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
