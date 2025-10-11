import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from utils.distributions import Categorical, DiagGaussian
from utils.model import get_grid, ChannelPool, Flatten, NNBase
import envs.utils.depth_utils as du
import envs.utils.pose as pu
from queue import Queue
import cv2
from skimage.draw import line


class Semantic_Mapping(nn.Module):

    """
    Semantic_Mapping
    """

    def __init__(self, args):
        super(Semantic_Mapping, self).__init__()

        self.device = args.device
        self.screen_h = args.frame_height
        self.screen_w = args.frame_width
        self.resolution = args.map_resolution
        self.z_resolution = args.map_resolution
        self.map_size_cm = args.map_size_cm // args.global_downscaling
        self.n_channels = 3
        self.vision_range = args.vision_range
        self.dropout = 0.5
        self.fov = args.hfov
        self.du_scale = args.du_scale
        self.cat_pred_threshold = args.cat_pred_threshold
        self.exp_pred_threshold = args.exp_pred_threshold
        self.map_pred_threshold = args.map_pred_threshold
        self.num_sem_categories = args.num_sem_categories

        self.max_height = int(200 / self.z_resolution)
        self.min_height = int(-40 / self.z_resolution)
        self.agent_height = args.camera_height * 100.
        self.shift_loc = [self.vision_range *
                          self.resolution // 2, 0, np.pi / 2.0]
        self.camera_matrix = du.get_camera_matrix(
            self.screen_w, self.screen_h, self.fov)

        self.pool = ChannelPool(1)

        vr = self.vision_range

        self.init_grid = torch.zeros(
            args.num_processes, 1 + self.num_sem_categories, vr, vr,
            self.max_height - self.min_height
        ).float().to(self.device)
        self.feat = torch.ones(
            args.num_processes, 1 + self.num_sem_categories,
            self.screen_h // self.du_scale * self.screen_w // self.du_scale
        ).float().to(self.device)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.stair_mask_radius = 30
        self.stair_mask = self.get_mask(self.stair_mask_radius).to(self.device)

    def forward(self, obs, pose_obs, maps_last, poses_last,eve_angle):
        bs, c, h, w = obs.size()
        depth = obs[:, 3, :, :]

        point_cloud_t = du.get_point_cloud_from_z_t(
            depth, self.camera_matrix, self.device, scale=self.du_scale)

        agent_view_t = du.transform_camera_view_t(
            point_cloud_t, self.agent_height, eve_angle, self.device)

        agent_view_centered_t = du.transform_pose_t(
            agent_view_t, self.shift_loc, self.device)

        max_h = self.max_height
        min_h = self.min_height
        xy_resolution = self.resolution
        z_resolution = self.z_resolution
        vision_range = self.vision_range
        XYZ_cm_std = agent_view_centered_t.float()
        XYZ_cm_std[..., :2] = (XYZ_cm_std[..., :2] / xy_resolution)
        XYZ_cm_std[..., :2] = (XYZ_cm_std[..., :2] -
                               vision_range // 2.) / vision_range * 2.
        XYZ_cm_std[..., 2] = XYZ_cm_std[..., 2] / z_resolution
        XYZ_cm_std[..., 2] = (XYZ_cm_std[..., 2] -
                              (max_h + min_h) // 2.) / (max_h - min_h) * 2.
        self.feat[:, 1:, :] = nn.AvgPool2d(self.du_scale)(
            obs[:, 4:, :, :]
        ).view(bs, c - 4, h // self.du_scale * w // self.du_scale)

        XYZ_cm_std = XYZ_cm_std.permute(0, 3, 1, 2)
        XYZ_cm_std = XYZ_cm_std.view(XYZ_cm_std.shape[0],
                                     XYZ_cm_std.shape[1],
                                     XYZ_cm_std.shape[2] * XYZ_cm_std.shape[3])

        voxels = du.splat_feat_nd(
            self.init_grid * 0., self.feat, XYZ_cm_std).transpose(2, 3)

        min_z = int(25 / z_resolution - min_h)
        max_z = int((self.agent_height +50) / z_resolution - min_h)
        mid_z = int(self.agent_height / z_resolution - min_h)

        agent_height_proj = voxels[..., min_z:max_z].sum(4)
        agent_height_stair_proj = voxels[..., mid_z-5:mid_z].sum(4)
        all_height_proj = voxels.sum(4)

        fp_map_pred = agent_height_proj[:, 0:1, :, :]
        fp_exp_pred = all_height_proj[:, 0:1, :, :]
        fp_stair_pred = agent_height_stair_proj[:, 0:1, :, :]
        fp_map_pred = fp_map_pred / self.map_pred_threshold
        fp_stair_pred = fp_stair_pred / self.map_pred_threshold
        fp_exp_pred = fp_exp_pred / self.exp_pred_threshold
        fp_map_pred = torch.clamp(fp_map_pred, min=0.0, max=1.0)
        fp_stair_pred = torch.clamp(fp_stair_pred, min=0.0, max=1.0)
        fp_exp_pred = torch.clamp(fp_exp_pred, min=0.0, max=1.0)

        pose_pred = poses_last

        agent_view = torch.zeros(bs, c,
                                 self.map_size_cm // self.resolution,
                                 self.map_size_cm // self.resolution
                                 ).to(self.device)

        x1 = self.map_size_cm // (self.resolution * 2) - self.vision_range // 2
        x2 = x1 + self.vision_range
        y1 = self.map_size_cm // (self.resolution * 2)
        y2 = y1 + self.vision_range
        agent_view[:, 0:1, y1:y2, x1:x2] = fp_map_pred
        agent_view[:, 1:2, y1:y2, x1:x2] = fp_exp_pred
        agent_view[:, 4:, y1:y2, x1:x2] = torch.clamp(
            agent_height_proj[:, 1:, :, :] / self.cat_pred_threshold,
            min=0.0, max=1.0)

        agent_view_stair = agent_view.clone().detach()
        agent_view_stair[:, 0:1, y1:y2, x1:x2] = fp_stair_pred
        corrected_pose = pose_obs


        def get_new_pose_batch(pose, rel_pose_change):

            pose[:, 1] += rel_pose_change[:, 0] * \
                torch.sin(pose[:, 2] / 57.29577951308232) \
                + rel_pose_change[:, 1] * \
                torch.cos(pose[:, 2] / 57.29577951308232)
            pose[:, 0] += rel_pose_change[:, 0] * \
                torch.cos(pose[:, 2] / 57.29577951308232) \
                - rel_pose_change[:, 1] * \
                torch.sin(pose[:, 2] / 57.29577951308232)
            pose[:, 2] += rel_pose_change[:, 2] * 57.29577951308232

            pose[:, 2] = torch.fmod(pose[:, 2] - 180.0, 360.0) + 180.0
            pose[:, 2] = torch.fmod(pose[:, 2] + 180.0, 360.0) - 180.0

            return pose

        current_poses = get_new_pose_batch(poses_last, corrected_pose)
        st_pose = current_poses.clone().detach()

        st_pose[:, :2] = - (st_pose[:, :2]
                            * 100.0 / self.resolution
                            - self.map_size_cm // (self.resolution * 2)) /\
            (self.map_size_cm // (self.resolution * 2))
        st_pose[:, 2] = 90. - (st_pose[:, 2])

        rot_mat, trans_mat = get_grid(st_pose, agent_view.size(),
                                      self.device)

        rotated = F.grid_sample(agent_view, rot_mat, align_corners=True)
        translated = F.grid_sample(rotated, trans_mat, align_corners=True)

        diff_ob_ex = translated[:, 1:2, :, :] - self.max_pool(translated[:, 0:1, :, :])

        diff_ob_ex[diff_ob_ex>0.8] = 1.0
        diff_ob_ex[diff_ob_ex!=1.0] = 0.0

        maps2 = torch.cat((maps_last.unsqueeze(1), translated.unsqueeze(1)), 1)

        map_pred, _ = torch.max(maps2, 1)

        for i in range(eve_angle.shape[0]):
            if eve_angle[i] == 0:
                map_pred[i, 0:1, :, :][diff_ob_ex[i] == 1.0] = 0.0

        # stairs view
        rot_mat_stair, trans_mat_stair = get_grid(st_pose, agent_view_stair.size(),
                                      self.device)

        rotated_stair = F.grid_sample(agent_view_stair, rot_mat_stair, align_corners=True)
        translated_stair = F.grid_sample(rotated_stair, trans_mat_stair, align_corners=True)

        stair_mask = torch.zeros(self.map_size_cm // self.resolution, self.map_size_cm // self.resolution).to(self.device)

        s_y = int(current_poses[0][1]*100/5)
        s_x = int(current_poses[0][0]*100/5)
        limit_up = self.map_size_cm // self.resolution - self.stair_mask_radius - 1
        limit_be = self.stair_mask_radius
        if s_y > limit_up:
            s_y = limit_up
        if s_y < self.stair_mask_radius:
            s_y = self.stair_mask_radius
        if s_x > limit_up:
            s_x = limit_up
        if s_x < self.stair_mask_radius:
            s_x = self.stair_mask_radius
        stair_mask[int(s_y-self.stair_mask_radius):int(s_y+self.stair_mask_radius), int(s_x-self.stair_mask_radius):int(s_x+self.stair_mask_radius)] = self.stair_mask

        translated_stair[0, 0:1, :, :] *= stair_mask
        translated_stair[0, 1:2, :, :] *= stair_mask

        # translated_stair[:, 13:14, :, :] = -self.max_pool(-translated_stair[:, 13:14, :, :])

        diff_ob_ex = translated_stair[:, 1:2, :, :] - translated_stair[:, 0:1, :, :]

        diff_ob_ex[diff_ob_ex>0.8] = 1.0
        diff_ob_ex[diff_ob_ex!=1.0] = 0.0

        maps3 = torch.cat((maps_last.unsqueeze(1), translated_stair.unsqueeze(1)), 1)

        map_pred_stair, _ = torch.max(maps3, 1)

        for i in range(eve_angle.shape[0]):
            if eve_angle[i] == 0:
                map_pred_stair[i, 0:1, :, :][diff_ob_ex[i] == 1.0] = 0.0


        return translated, map_pred, map_pred_stair, current_poses

    def get_mask(self, step_size):
        size = int(step_size) * 2 
        mask = torch.zeros(size, size)
        for i in range(size):
            for j in range(size):
                if ((i + 0.5) - (size // 2)) ** 2 + \
                ((j + 0.5) - (size // 2)) ** 2 <= \
                        step_size ** 2:
                    mask[i, j] = 1
        return mask

def preprocess_exp_map(exp_map,b_obs_map,pose_pred):
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(20, 20))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
    exp_map=cv2.dilate(exp_map.astype(np.float32),kernel2)
    # obs_map = cv2.dilate(obs_map, kernel)
        # 对障碍物反的膨胀进行骨架提取
    obs_map = np.where(b_obs_map == 0, 1, 0)
    m, n = exp_map.shape
    visited = np.zeros_like(exp_map, dtype=bool)
    start_x, start_y, start_o, gx1, gx2, gy1, gy2 = \
            pose_pred
    gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
    r, c = start_y, start_x
    start = [int(r * 100.0 / 5 - gx1),
                int(c * 100.0 / 5 - gy1)]
    start = pu.threshold_poses(start, exp_map.shape)
    mask,dist = fov_mask(exp_map.copy(),start[0],start[1],np.deg2rad(-start_o),np.pi*80/180,40)
    queue = Queue()
    queue.put(start)
    visited[start[0],start[1]] = True
    view_mask = np.zeros_like(exp_map, dtype=bool)
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0),\
                (-1,-1),(1,1),(-1,1),(1,-1)]  # 上下左右八个方向
    while not queue.empty():
        x, y = queue.get()
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (0 <= nx < m) and (0 <= ny < n) and  mask[nx, ny]:
                if (not visited[nx, ny]) and (  obs_map[nx,ny]):
                    visited[nx, ny] = True
                    queue.put((nx, ny))
                    view_mask[nx, ny] = True
    old_mask = np.zeros_like(exp_map, dtype=bool)#部分遮挡fov
    map_in_fov=np.logical_and(mask,b_obs_map)
    while dist.sum()!=0:
        x,y=np.where(dist==dist.max())
        rr,cc=line(start[0],start[1],x[0],y[0])
        line1=np.zeros_like(exp_map, dtype=bool)
        line1[rr,cc]=1
        if np.logical_and(line1,map_in_fov).sum()==0:
            old_mask[line1]=True
            dist[line1]=0
        else:
            dist[x,y]=0





    
    return view_mask

def fov_mask(arr, y, x, fov_deg, fov_bearing,fov_range):
    """
    生成视野范围内的掩码
    """
    # 计算视野范围的角度范围
    min_theta = fov_deg - fov_bearing / 2
    max_theta = fov_deg + fov_bearing / 2

    # 计算每个像素到机器人的距离和角度
    y_idx, x_idx = np.indices(arr.shape)
    delta_x = x_idx - x
    delta_y = y_idx - y
    dist = np.sqrt(delta_x**2 + delta_y**2)
    angle = -np.arctan2(delta_y, delta_x)
    if max_theta > np.pi:
        mask = ((angle >= min_theta) | (angle <= max_theta-2*np.pi)) & (dist <= fov_range)
    elif min_theta < -np.pi:
        mask = ((angle >= min_theta+2*np.pi) | (angle <= max_theta)) & (dist <= fov_range)
    # 计算掩码
    else:
        mask = (angle >= min_theta) & (angle <= max_theta) & (dist <= fov_range)
    dist=dist*mask
    return mask.astype(np.bool),dist.astype(np.float32)
