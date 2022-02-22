import numpy as np
from os import path as osp

import pickle
from mmdet3d.core import show_result
from mmdet3d.core.bbox import DepthInstance3DBoxes
from mmdet3d.datasets.custom_3d import Custom3DDataset
from mmdet.datasets import DATASETS


@DATASETS.register_module()
class WayveDataset(Custom3DDataset):
    CLASSES = ('vehicle', 'pedestrian', 'bicycle', 'motorcycle', 'cyclist', 'motorcyclist', 'unknown')

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations.
        """
        with open(ann_file, 'rb') as f:
            return pickle.load(f)

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        import ipdb; ipdb.set_trace()
        info = self.data_infos[index]
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['sample_idx'],
            pts_filename=info['pts_filename'],
            sweeps=info['sweeps'],
            timestamp=info['timestamp_us'] / 1e6,
        )

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            for cam_type, cam_info in info['cameras'].items():
                image_paths.append(cam_info['path'])
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)

                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt.T)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                ))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        return input_dict

    def get_ann_info(self, index):
        # Use index to get the annos, thus the evalhook could also use this api
        info = self.data_infos[index]
        if info['annos']['gt_num'] != 0:
            gt_bboxes_3d = info['annos']['gt_boxes_upright_depth'].astype(
                np.float32)  # k, 6
            gt_labels_3d = info['annos']['class'].astype(np.long)
        else:
            gt_bboxes_3d = np.zeros((0, 6), dtype=np.float32)
            gt_labels_3d = np.zeros((0, ), dtype=np.long)

        # to target box structure
        gt_bboxes_3d = DepthInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            with_yaw=False,
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        pts_instance_mask_path = osp.join(self.data_root,
                                          info['pts_instance_mask_path'])
        pts_semantic_mask_path = osp.join(self.data_root,
                                          info['pts_semantic_mask_path'])

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            pts_instance_mask_path=pts_instance_mask_path,
            pts_semantic_mask_path=pts_semantic_mask_path)
        return anns_results
