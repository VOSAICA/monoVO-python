import torch

from .superpoint import SuperPoint
from .superglue import SuperGlue
from .nearest_neighbor_matcher import NearestNeighborMatcher


class SuperpointDetector(torch.nn.Module):
    """ Image Matching Frontend (SuperPoint + SuperGlue) """
    def __init__(self, config={}):
        super().__init__()
        self.superpoint = SuperPoint(config.get('superpoint', {}))
        if config.get('use_nn_matcher', False):
            self.matcher = NearestNeighborMatcher(config.get('nn', {}))
        else:
            self.matcher = SuperGlue(config.get('superglue', {}))

    def forward(self, data):
        """ Run SuperPoint (optionally) and SuperGlue
        SuperPoint is skipped if ['keypoints0', 'keypoints1'] exist in input
        Args:
          data: dictionary with minimal keys: ['image0', 'image1']
        """
        pred = {}

        # Extract SuperPoint (keypoints, scores, descriptors) if not provided

        pred = {**self.superpoint({'image': data['image']})}

        return pred
