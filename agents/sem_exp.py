import math
import os
import cv2
import numpy as np
import skimage.morphology
from PIL import Image
from torchvision import transforms
import torch
import torch.nn as nn
from envs.utils.fmm_planner import FMMPlanner
from envs.habitat.objectgoal_env import ObjectGoal_Env_HM3D
from agents.utils.semantic_prediction import SemanticPredMaskRCNN
from constants import color_palette,coco_categories_inv
import envs.utils.pose as pu
import agents.utils.visualization as vu
import time
from queue import Queue
import heapq
import networkx as nx
from skimage import measure
from skimage.graph import route_through_array
import openai
from openai import OpenAI

API_SECRET_KEY = "814af94051162c79e69d5dec40bc1736";
BASE_URL = "https://flag.smarttrot.com/v1/"
client = OpenAI(api_key=API_SECRET_KEY, base_url=BASE_URL)

class Sem_Exp_Env_Agent(ObjectGoal_Env_HM3D):
    """The Sem_Exp environment agent class. A seperate Sem_Exp_Env_Agent class
    object is used for each environment thread.

    """

    def __init__(self, args, rank, config_env, dataset):

        self.args = args
        super().__init__(args, rank, config_env, dataset)

        # initialize transform for RGB observations
        self.res = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.Resize((args.frame_height, args.frame_width),
                               interpolation=Image.NEAREST)])

        # initialize semantic segmentation prediction model
        if args.sem_gpu_id == -1:
            args.sem_gpu_id = config_env.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID

        self.sem_pred = SemanticPredMaskRCNN(args)

        # initializations for planning:
        self.selem = skimage.morphology.disk(3)

        self.obs = None
        self.obs_shape = None
        self.collision_map = None
        self.visited = None
        self.visited_vis = None
        self.col_width = None
        self.curr_loc = None
        self.last_loc = None
        self.last_action = None
        self.count_forward_actions = None
        self.last_goal =np.zeros((args.map_size_cm // args.map_resolution//args.global_downscaling,
                     args.map_size_cm // args.map_resolution//args.global_downscaling))
        self.seleted_goal= None
        self.next_node = None
        self.fail_case = {}
        self.fail_case['collision'] = 0
        self.fail_case['success'] = 0
        self.fail_case['detection'] = 0
        self.fail_case['exploration'] = 0

        self.collision_number=0
        self.forward_number=1
        self.explored_area=0
        self.path_legnth=1
        self.filename=0
        
        if args.visualize or args.print_images:
            self.legend = cv2.imread('docs/legend.png')
            self.vis_image = None
            self.rgb_vis = None

    def reset(self):
        args = self.args

        obs, info = super().reset()
        obs = self._preprocess_obs(obs)

        self.obs_shape = obs.shape

        # Episode initializations
        map_shape = (args.map_size_cm // args.map_resolution,
                     args.map_size_cm // args.map_resolution)
        self.collision_map = np.zeros(map_shape)
        self.visited = np.zeros(map_shape)
        self.visited_vis = np.zeros(map_shape)
        self.col_width = 1
        self.count_forward_actions = 0
        self.curr_loc = [args.map_size_cm / 100.0 / 2.0,
                         args.map_size_cm / 100.0 / 2.0, 0.]
        self.last_action = None
        self.cflag = 0 #机器人的状态，0表示刚开始，1代表到达节点处，2代表需要选点， 3代表在移动过程中,4是刚选过点还没离开节点
        self.rotate_count = 0 #在节点处旋转的次数
        self.last_window= None
        self.this_window = None
        self.fmm_goal=None
        self.info=info
        self.stuck_count=0
        self.last_prompt=""
        self.last_answer=""
        self.last_path_semantic=[]
        self.path_mask=None
        self.joint_visited=np.zeros(map_shape)
        self.caption=""
        self.orientation_caption={}
        self.caption_locations=[]
        self.replan_count=0
        self.find_collision=0
        self.down_flag=0
        self.eve_angle = 0
        self.eve_angle_old = 0
        self.replan_count=0
        self.local_random_count=0
        info['eve_angle'] = self.eve_angle
        self.collision_number=0
        self.forward_number=1
        self.explored_area=0
        self.path_legnth=1
        if args.visualize or args.print_images:
            self.vis_image = vu.init_vis_image(self.goal_name, self.legend)
        return obs, info

    def new_episodes(self):
        args = self.args
        map_shape = (args.map_size_cm // args.map_resolution,
        args.map_size_cm // args.map_resolution)
        self.collision_map = np.zeros(map_shape)
        self.visited = np.zeros(map_shape)
        self.visited_vis = np.zeros(map_shape)
        self.cflag=0
        self.col_width = None
        self.last_loc = None
        self.last_action = None
        self.count_forward_actions = None
        self.last_goal =np.zeros((args.map_size_cm // args.map_resolution//args.global_downscaling,
                     args.map_size_cm // args.map_resolution//args.global_downscaling))
        self.seleted_goal= None
        self.next_node = None
        self.col_width = 1
        self.count_forward_actions = 0
        self.curr_loc = [args.map_size_cm / 100.0 / 2.0,
                         args.map_size_cm / 100.0 / 2.0, 0.]
        self.last_action = None
        self.cflag = 0 #机器人的状态，0表示刚开始，1代表到达节点处，2代表需要选点， 3代表在移动过程中,4是刚选过点还没离开节点
        self.rotate_count = 0 #在节点处旋转的次数
        self.last_window= None
        self.this_window = None
        self.last_prompt=""
        self.last_answer=""
        self.last_path_semantic=[]
        self.joint_visited=np.zeros(map_shape)
        self.caption=""
        self.orientation_caption={}
        self.caption_locations=[]
        self.find_collision=0
        self.stuck_count = 0
        self.down_flag=0
        self.replan_count=0
        self.local_random_count=0
        if args.visualize or args.print_images:
            self.vis_image = vu.init_vis_image(self.goal_name, self.legend)
        self.collision_number=0
        self.forward_number=1
        self.explored_area=0
        self.path_legnth=1

    def plan_act_and_preprocess(self, planner_inputs):
        """Function responsible for planning, taking the action and
        preprocessing observations

        Args:
            planner_inputs (dict):
                dict with following keys:
                    'map_pred'  (ndarray): (M, M) map prediction
                    'goal'      (ndarray): (M, M) mat denoting goal locations
                    'pose_pred' (ndarray): (7,) array denoting pose (x,y,o)
                                 and planning window (gx1, gx2, gy1, gy2)
                     'found_goal' (bool): whether the goal object is found

        Returns:
            obs (ndarray): preprocessed observations ((4+C) x H x W)
            reward (float): amount of reward returned after previous action
            done (bool): whether the episode has ended
            info (dict): contains timestep, pose, goal category and
                         evaluation metric info
        """

        if planner_inputs["clr"]==1:
            self.new_episodes()

        if planner_inputs["new_goal"]:
            self.info['clear_flag'] = 0

        # plan
        if planner_inputs["wait"]:
            self.last_action = None
            self.info["sensor_pose"] = [0., 0., 0.]
            return np.zeros(self.obs.shape), 0., False, self.info

        action = self._plan(planner_inputs)
        print("stuck_count:"+str(self.stuck_count)+"replan_count:"+str(self.replan_count))

        if self.stuck_count > 20 or self.replan_count > 35:
            self.info['clear_flag'] = 1
            self.stuck_count = 0
            self.cflag=0
            self.rotate_count=0
            self.joint_visited=np.zeros_like(self.joint_visited)
            self.orientation_caption={}
            self.caption_locations=[]

        if action >= 0:

            # act
            action = {'action': action}
            if action==1:
                self.forward_number+=1

            obs, rew, done, info = super().step(action)

            if done and self.info['success'] == 0:
                if self.info['time'] >= self.args.max_episode_length - 1:
                    self.fail_case['exploration'] += 1
                elif self.replan_count>35 or planner_inputs['found_goal']==1:
                    self.fail_case['collision'] += 1
                else:
                    self.fail_case['detection'] += 1

            if done and self.info['success'] == 1:
                self.fail_case['success'] += 1
            # preprocess obs
            obs = self._preprocess_obs(obs) 
            self.last_action = action['action']
            self.obs = obs
            self.info = info
            info['eve_angle'] = self.eve_angle
            self.info['fail_case']=self.fail_case
            self.info['episode_id']=self.episode_no
            self.info['collision_number']=self.collision_number
            self.info['forward_number']=self.forward_number
            self.info['explored_area']=np.sum(np.rint(planner_inputs['exp_pred']))
            self.info['path_length']=0.25*self.forward_number
            return obs, rew, done, info
        else:
            self.last_action = None
            self.info["sensor_pose"] = [0., 0., 0.]
            return np.zeros(self.obs_shape), 0., False, self.info

    def _plan(self, planner_inputs):
        """Function responsible for planning

        Args:
            planner_inputs (dict):
                dict with following keys:
                    'map_pred'  (ndarray): (M, M) map prediction
                    'goal'      (ndarray): (M, M) goal locations
                    'pose_pred' (ndarray): (7,) array  denoting pose (x,y,o)
                                 and planning window (gx1, gx2, gy1, gy2)
                    'found_goal' (bool): whether the goal object is found

        Returns:
            action (int): action id
        """

        time1=time.perf_counter()

        args = self.args
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = \
            planner_inputs['pose_pred']
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]
        self.this_window = planning_window
        r, c = start_y, start_x
        start = [int(r * 100.0 / args.map_resolution - gx1),
                 int(c * 100.0 / args.map_resolution - gy1)]
        map_pred = np.rint(planner_inputs['map_pred'])
        start = pu.threshold_poses(start, map_pred.shape)
        exp_pred = np.rint(planner_inputs['exp_pred'])

        angle_agent = (start_o) % 360.0
        if angle_agent > 180:
            angle_agent -= 360

        if self.last_window is None:
            self.last_window = planning_window
        if (planner_inputs['found_goal']!= 1 )and self.last_window!=self.this_window and self.next_node is not None:##相当于进行了一个预处理操作，如果窗口变了，就把上一个窗口的goal进行相应的偏移
            self.next_node=(self.next_node[0]+self.last_window[0]-self.this_window[0],self.next_node[1]+self.last_window[2]-self.this_window[2])
            self.last_window = planning_window
            map_pred = np.rint(planner_inputs['map_pred'])
            goal = area_mask(map_pred,self.next_node[0],self.next_node[1],2)
            self.last_goal=goal

        self.last_loc = self.curr_loc

        if planner_inputs['found_goal'] == 1:
            self.rotate_count=0
            goal = planner_inputs['goal']
            ###这里应该考虑电视机
            if self.info['goal_cat_id'] == 5:
                selem = skimage.morphology.disk(15)
            else:
                selem = skimage.morphology.disk(6)
            goal_mat = 1 - skimage.morphology.binary_dilation(
                goal, selem) != True
            raw_goal = goal_mat == 1
            img_label,num=measure.label(raw_goal,connectivity=2,return_num=True)
            max_num=1
            for i in range(num):
                if np.sum(img_label==i+1)>np.sum(img_label==max_num):
                    max_num=i+1
            goal=img_label==max_num
            map_pred = np.rint(planner_inputs['map_pred'])
            _,_,_,_,_,_=self._visualize(planner_inputs.copy(),2)
        else:
            #找到未知最大的voronoi结点
            ## add the evelution angle
            eve_start_x = int(5 * math.sin(angle_agent/180*np.pi) + start[0])
            eve_start_y = int(5 * math.cos(angle_agent/180*np.pi) + start[1])
            if eve_start_x > map_pred.shape[0]: eve_start_x = map_pred.shape[0] 
            if eve_start_y > map_pred.shape[0]: eve_start_y = map_pred.shape[0] 
            if eve_start_x < 0: eve_start_x = 0 
            if eve_start_y < 0: eve_start_y = 0 
            if exp_pred[eve_start_x, eve_start_y] == 0 and self.eve_angle > -60:
                action = 5
                self.eve_angle -= 30
                self._visualize(planner_inputs.copy(),self.cflag)
                return action
            
            elif exp_pred[eve_start_x, eve_start_y] == 1 and self.eve_angle < 0:
                action = 4
                self.eve_angle += 30
                self._visualize(planner_inputs.copy(),self.cflag)
                return action

            exploratory_node,adjacent_node,far_node,skeleton_mask,joint_node,voronoi_graph=self._visualize(planner_inputs.copy(),self.cflag)

            if self.cflag == 0:

                joint_mask= np.zeros_like(self.last_goal)

                if self.rotate_count == 0 and (self.joint_visited[gx1:gx2, gy1:gy2][start[0],start[1]]==1) :
                    self.rotate_count =12                
                else:
                    #print("ssssssssssssssssss")
                    if round(start_o/30)*30==180:
                        tmp_start_o=-180
                    else:
                        tmp_start_o=round(start_o/30)*30
                    if self.rotate_count == 0:
                        self.orientation_caption[start[0]+gx1,start[1]+gy1]={tmp_start_o:self.caption}
                        self.caption_locations.append([start[0]+gx1,start[1]+gy1])
                    else:
                        try:
                            self.orientation_caption[start[0]+gx1,start[1]+gy1][tmp_start_o]=self.caption
                        except:
                            near_idx=0
                            near_value=100000
                            for i in range(len(self.caption_locations)):
                                if (self.caption_locations[i][0]-start[0]-gx1)**2+(self.caption_locations[i][1]-start[1]-gy1)**2<near_value:
                                    near_idx=i
                                    near_value=(self.caption_locations[i][0]-start[0]-gx1)**2+(self.caption_locations[i][1]-start[1]-gy1)**2
                            self.orientation_caption[self.caption_locations[near_idx][0],self.caption_locations[near_idx][1]][tmp_start_o]=self.caption

                self.rotate_count += 1
                action = 3
                goal = np.zeros((map_pred.shape[0],map_pred.shape[1]))
                if self.rotate_count == 13:
                    self.cflag = 2 #需要需要选点
                    self.rotate_count = 0
                    self.joint_visited[gx1:gx2, gy1:gy2] = \
                        vu.draw_line(start, start, self.joint_visited[gx1:gx2, gy1:gy2],25,8)
                else:
                    self.last_goal=goal
                    return action

            if list(adjacent_node.values())[0]<8**2:
                if self.cflag == 3:
                    self.cflag=0
            if self.cflag == 3 and self.local_random_count>0:
                self.local_random_count+=1
                if self.local_random_count==10:
                    self.cflag=2
                    self.local_random_count=0

            if self.cflag == 4 and list(adjacent_node.values())[0]>9**2: ##选过点后
                self.cflag =3

            if self.cflag == 2: ##选点
                if len(exploratory_node)==0:
                    actions = torch.randn(1, 2)*6
                    cpu_actions = nn.Sigmoid()(actions).numpy()
                    global_goals = [[int(action[0] * 240),
                                        int(action[1] * 240)]
                                    for action in cpu_actions]
                    global_goals = [[min(x, int(240 - 1)),
                                        min(y, int(240 - 1))]
                                    for x, y in global_goals][0]
                    self.next_node = (global_goals[0],global_goals[1])
                    self.cflag=4
                    self.local_random_count=1
                    
                else:
                    my_node = list(adjacent_node.keys())[0]
                    self.next_node = self.best_node(my_node, exploratory_node, skeleton_mask.copy(), joint_node[:], planner_inputs.copy(),self.info['goal_cat_id'],voronoi_graph)
                    self.cflag = 4

            # Get Map prediction
            map_pred = np.rint(planner_inputs['map_pred'])
            goal = area_mask(map_pred,self.next_node[0],self.next_node[1],1)

        self.last_goal=goal

        # Get curr loc
        self.curr_loc = [start_x, start_y, start_o]
        r, c = start_y, start_x
        start = [int(r * 100.0 / args.map_resolution - gx1),
                 int(c * 100.0 / args.map_resolution - gy1)]
        start = pu.threshold_poses(start, map_pred.shape)

        self.visited[gx1:gx2, gy1:gy2][start[0] - 0:start[0] + 1,
                                       start[1] - 0:start[1] + 1] = 1

        if args.visualize or args.print_images:
            # Get last loc
            last_start_x, last_start_y = self.last_loc[0], self.last_loc[1]
            r, c = last_start_y, last_start_x
            last_start = [int(r * 100.0 / args.map_resolution - gx1),
                          int(c * 100.0 / args.map_resolution - gy1)]
            last_start = pu.threshold_poses(last_start, map_pred.shape)
            self.visited_vis[gx1:gx2, gy1:gy2] = \
                vu.draw_line(last_start, start, self.visited_vis[gx1:gx2, gy1:gy2])

        # Collision check
        if self.last_action == 1:
            x1, y1, t1 = self.last_loc
            x2, y2, _ = self.curr_loc
            buf = 4
            length = 2

            if abs(x1 - x2) < 0.05 and abs(y1 - y2) < 0.05:
                self.col_width += 2
                if self.col_width == 7:
                    length = 4
                    buf = 3
                self.col_width = min(self.col_width, 5)
            else:
                self.col_width = 1

            dist = pu.get_l2_distance(x1, x2, y1, y2)
            if dist < args.collision_threshold:  # Collision
                if planner_inputs['found_goal'] == 1:
                    self.find_collision +=1
                else:
                    self.collision_number+=1
                self.stuck_count +=1
                width = self.col_width

                for i in range(length):
                    for j in range(width):
                        wx = x1 + 0.05 * \
                            ((i + buf) * np.cos(np.deg2rad(t1))
                             + (j - width // 2) * np.sin(np.deg2rad(t1)))
                        wy = y1 + 0.05 * \
                            ((i + buf) * np.sin(np.deg2rad(t1))
                             - (j - width // 2) * np.cos(np.deg2rad(t1)))
                        r, c = wy, wx
                        r, c = int(r * 100 / args.map_resolution), \
                            int(c * 100 / args.map_resolution)
                        [r, c] = pu.threshold_poses([r, c],
                                                    self.collision_map.shape)
                        self.collision_map[r, c] = 1

            else:
                if planner_inputs['found_goal'] != 1:
                    self.find_collision =0        
                self.stuck_count = 0        
                width = self.col_width
                buf=2
                length=6
                width=6
                for i in range(length):
                    for j in range(width):
                        wx = x1 + 0.05 * \
                            ((i + buf) * np.cos(np.deg2rad(t1))
                             + (j - width // 2) * np.sin(np.deg2rad(t1)))
                        wy = y1 + 0.05 * \
                            ((i + buf) * np.sin(np.deg2rad(t1))
                             - (j - width // 2) * np.cos(np.deg2rad(t1)))
                        r, c = wy, wx
                        r, c = int(r * 100 / args.map_resolution), \
                            int(c * 100 / args.map_resolution)
                        [r, c] = pu.threshold_poses([r, c],
                                                    self.collision_map.shape)
                        self.collision_map[r, c] = 0

        if np.sum(goal)>0:
            stg, replan,stop = self._get_stg(map_pred, start, np.copy(goal),
                                  planning_window,planner_inputs['found_goal'])
            self.fmm_stop=stop
            if replan:
                self.replan_count += 1
                print("false: ", self.replan_count)
            else:
                self.replan_count = 0
        else:
            action=3
            self.cflag=0
            return action
        time2=time.perf_counter()

        # Deterministic Local Policy
        if (stop and planner_inputs['found_goal'] == 1) or self.replan_count > 35:
            action = 0  # Stop
        elif stop:
            action=3
            self.cflag=0
        else:
            (stg_x, stg_y) = stg
            self.fmm_goal=area_mask(map_pred,stg[0],stg[1],2)
            angle_st_goal = math.degrees(math.atan2(stg_x - start[0],
                                                    stg_y - start[1]))
            angle_agent = (start_o) % 360.0
            if angle_agent > 180:
                angle_agent -= 360

            relative_angle = (angle_agent - angle_st_goal) % 360.0
            if relative_angle > 180:
                relative_angle -= 360

            if relative_angle > self.args.turn_angle / 2.:
                action = 3  # Right
            elif relative_angle < -self.args.turn_angle / 2.:
                action = 2  # Left
            else:
                action = 1  # Forward

        return action

    def _get_stg(self, grid, start, goal, planning_window,f_flag):
        """Get short-term goal"""

        [gx1, gx2, gy1, gy2] = planning_window
        x1, y1, = 0, 0
        x2, y2 = grid.shape

        def add_boundary(mat, value=1):
            h, w = mat.shape
            new_mat = np.zeros((h + 2, w + 2)) + value
            new_mat[1:h + 1, 1:w + 1] = mat
            return new_mat

        traversible = skimage.morphology.binary_dilation(
            grid[x1:x2, y1:y2], self.selem) != True
        traversible[self.collision_map[gx1:gx2, gy1:gy2]
                    [x1:x2, y1:y2] == 1] = 0
        traversible[self.visited[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 1

        traversible[int(start[0] - x1) - 1:int(start[0] - x1) + 2,
                    int(start[1] - y1) - 1:int(start[1] - y1) + 2] = 1

        traversible = add_boundary(traversible)
        goal = add_boundary(goal, value=0)

        planner = FMMPlanner(traversible)
        selem = skimage.morphology.disk(1)
        goal = skimage.morphology.binary_dilation(goal, selem) != True
        goal = 1 - goal * 1.
        planner.set_multi_goal(goal)

        state = [start[0] - x1 + 1, start[1] - y1 + 1]
        stg_x, stg_y, replan, stop = planner.get_short_term_goal(state)

        stg_x, stg_y = stg_x + x1 - 1, stg_y + y1 - 1

        return (stg_x, stg_y),replan, stop

    def _preprocess_obs(self, obs, use_seg=True):
        args = self.args
        obs = obs.transpose(1, 2, 0)
        rgb = obs[:, :, :3]
        depth = obs[:, :, 3:4]
        semantic = obs[:,:,4:5].squeeze()

        self.rgb_vis = rgb[:, :, ::-1]
        sem_seg_pred = np.zeros((rgb.shape[0], rgb.shape[1], 15 + 1))
        for i in range(16):
            sem_seg_pred[:,:,i][semantic == i+1] = 1
        self.caption = self.sem_pred.get_prediction(rgb)
        depth = self._preprocess_depth(depth, args.min_depth, args.max_depth)

        ds = args.env_frame_width // args.frame_width  # Downscaling factor
        if ds != 1:
            rgb = np.asarray(self.res(rgb.astype(np.uint8)))
            depth = depth[ds // 2::ds, ds // 2::ds]
            sem_seg_pred = sem_seg_pred[ds // 2::ds, ds // 2::ds]

        depth = np.expand_dims(depth, axis=2)
        state = np.concatenate((rgb, depth, sem_seg_pred),
                               axis=2).transpose(2, 0, 1)

        return state

    def _preprocess_depth(self, depth, min_d, max_d):
        depth = depth[:, :, 0] * 1

        for i in range(depth.shape[1]):
            depth[:, i][depth[:, i] == 0.] = depth[:, i].max()

        mask2 = depth > 0.99
        depth[mask2] = 0.

        mask1 = depth == 0
        depth[mask1] = 100.0
        depth = min_d * 100.0 + depth * (max_d-min_d) * 100.0
        return depth

    def _get_sem_pred(self, rgb, use_seg=True):
        if use_seg:
            semantic_pred, self.rgb_vis, self.caption = self.sem_pred.get_prediction(rgb)
            semantic_pred = semantic_pred.astype(np.float32)
            self.rgb_vis = rgb[:, :, ::-1]
        else:
            semantic_pred = np.zeros((rgb.shape[0], rgb.shape[1], 16))
            self.rgb_vis = rgb
        return semantic_pred

    def _visualize(self, inputs,search_mode):
        args = self.args
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))
        kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT,(16,16))

        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = inputs['pose_pred']
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        map_pred = np.rint(inputs['map_pred'].copy())
        frontier = inputs['map_edge'].copy()
        back_map_pred = cv2.dilate(map_pred, kernel)

        edge_mask = frontier == 1
        exp_pred = inputs['exp_pred'].copy()
        goal = inputs['goal'] if inputs['found_goal'] else self.last_goal

        sem_map = inputs['sem_map_pred'].copy()
        sem_map += 5

        last_start_x, last_start_y = self.last_loc[0], self.last_loc[1]
        r, c = last_start_y, last_start_x
        last_start = [int(r * 100.0 / args.map_resolution - gx1),
                        int(c * 100.0 / args.map_resolution - gy1)]
        last_start = pu.threshold_poses(last_start, map_pred.shape)
        true_exp_pred= exp_pred.copy()

        kernel = np.ones((10, 10), dtype=np.uint8)
        exp_pred = cv2.morphologyEx(exp_pred, cv2.MORPH_CLOSE, kernel)
        exp_pred=np.rint(exp_pred)==1
        voronoi = np.where(np.subtract(exp_pred,back_map_pred)==-1,0,np.subtract(exp_pred,back_map_pred))
        skeleton = skimage.morphology.skeletonize(voronoi)

        skeleton_pred = skeleton.astype(np.int)
        vis_mask = self.visited_vis[gx1:gx2, gy1:gy2] == 1
        no_cat_mask = sem_map == 20
        #得到探索和障碍物区域的mask
        map_mask = np.rint(map_pred) == 1
        exp_mask = np.rint(exp_pred) == 1
        true_exp_mask = np.rint(true_exp_pred) == 1
        skeleton_mask = np.rint(skeleton_pred) == 1
        
        # 找到最大的骨架
        img_label,num=measure.label(skeleton_mask,connectivity=2,return_num=True)
        max_num=1
        for i in range(num):
            if np.sum(img_label==i+1)>np.sum(img_label==max_num):
                max_num=i+1
        skeleton_mask=img_label==max_num
        
        # 生成交叉点
        old_joint_mask= np.zeros_like(sem_map)
        sem_map[no_cat_mask] = 0
        joint_mask= np.zeros_like(sem_map)
        for i in  np.argwhere(skeleton_mask):
            if np.sum(skeleton_mask[i[0]-1:i[0]+2, i[1]-1:i[1]+2]) >= 4:
                old_joint_mask[i[0]][i[1]]=1

        # 消除很短的枝岔
        for i in np.argwhere(skeleton_mask):
            if (np.sum(skeleton_mask[i[0]-1:i[0]+2, i[1]-1:i[1]+2]) == 2):
                tmp_exp_node_mask=np.zeros_like(sem_map)
                tmp_exp_node_mask[i[0]-4:i[0]+5,i[1]-4:i[1]+5]=1
                if np.sum(np.logical_and(tmp_exp_node_mask,old_joint_mask))==0:
                    old_joint_mask[i[0]][i[1]]=1

        for i in  np.argwhere(old_joint_mask):
            if i[0]<args.map_size_cm // args.map_resolution//args.global_downscaling-1 and i[0]>0 and i[1]<args.map_size_cm // args.map_resolution//args.global_downscaling-1 and i[1]>0:
                joint_mask[i[0]][i[1]]=old_joint_mask[i[0]-1, i[1]]+\
                                        old_joint_mask[i[0]+1, i[1]]+\
                                        old_joint_mask[i[0], i[1]-1]+\
                                        old_joint_mask[i[0], i[1]+1]+\
                                        old_joint_mask[i[0], i[1]]
            
        joint_node=[]
        for i in np.argwhere(joint_mask):
            if joint_mask[i[0]][i[1]]==np.max(joint_mask[i[0]-1:i[0]+2, i[1]-1:i[1]+2])\
                and joint_mask[i[0]][i[1]]>0:
                joint_node.append(i)
                joint_mask[i[0]-1:i[0]+2, i[1]-1:i[1]+2]=0

        # 开始建graph
        voronoi_graph=build_graph(skeleton_mask.copy(), joint_node.copy(),vis_mask.copy(),true_exp_mask.copy(),map_mask.copy())
        adjacent_node,_ = search_adjacent(last_start,joint_node,map_pred)
        far_node,_ = search_far(last_start,joint_node)

        m1=np.logical_and(no_cat_mask,true_exp_mask)
        sem_map[m1] = 2
        map_mask = np.rint(map_mask) == 1
        m2=np.logical_and(map_mask,no_cat_mask)
        sem_map[m2] = 1
        sem_map[skeleton_mask] = 4

        selem = skimage.morphology.disk(3)
        goal_mat = 1 - skimage.morphology.binary_dilation(
            goal, selem) != True
        
        goal_mask = goal_mat == 1
        sem_map[goal_mask] = 3
        sem_map[self.collision_map[gx1:gx2, gy1:gy2] == 1]= 3
        tmp_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(13,13))
        dilated_frontier = cv2.dilate(frontier.astype(np.float32), tmp_kernel)
        exploratory_node = {}
        for i in range(len(joint_node)):
            if dilated_frontier[joint_node[i][0],joint_node[i][1]]==1:
                exploratory_node[tuple(joint_node[i])]=1

        color_pal = [int(x * 255.) for x in color_palette]
        sem_map_vis = Image.new("P", (sem_map.shape[1],
                                        sem_map.shape[0]))
        sem_map_vis.putpalette(color_pal)
        sem_map_vis.putdata(sem_map.flatten().astype(np.uint8))
        sem_map_vis = sem_map_vis.convert("RGB")
        sem_map_vis = np.flipud(sem_map_vis)

        sem_map_vis = sem_map_vis[:, :, [2, 1, 0]]
        sem_map_vis = cv2.resize(sem_map_vis, (480, 480), interpolation=cv2.INTER_NEAREST)
        
        self.vis_image[50:530, 670:1150] = sem_map_vis
        self.vis_image[50:530, 15:655] = self.rgb_vis
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (20, 20, 20)  # BGR
        thickness = 3
        if search_mode not in [0,1,2]:
            search_mode=1
        mode=["Looking around","Navigating","Go for target"]
        text = "Navigation mode: "+mode[search_mode]
        textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
        textX = 30
        textY = (50 +1320+ textsize[1]) // 2
        color = (255,255,255)
        self.vis_image = cv2.putText(self.vis_image,"Navigation mode: Looking around", (textX, textY),
                                font, fontScale, color, thickness,
                                cv2.LINE_AA)
        self.vis_image = cv2.putText(self.vis_image, "Navigation mode: Navigating", (textX, textY),
                                font, fontScale, color, thickness,
                                cv2.LINE_AA)
        self.vis_image = cv2.putText(self.vis_image, "Navigation mode: Go for target", (textX, textY),
                                font, fontScale, color, thickness,
                                cv2.LINE_AA)
        color = (255,0,0)
        thickness=2
        self.vis_image = cv2.putText(self.vis_image, text, (textX, textY),
                                font, fontScale, color, thickness,
                                cv2.LINE_AA)

        pos = (
            (start_x * 100. / args.map_resolution - gy1)
            * 480 / map_pred.shape[0],
            (map_pred.shape[1] - start_y * 100. / args.map_resolution + gx1)
            * 480 / map_pred.shape[1],
            np.deg2rad(-start_o)
        )
        
        agent_arrow = vu.get_contour_points(pos, origin=(670, 50))
        color = (int(color_palette[11] * 255),
                int(color_palette[10] * 255),
                int(color_palette[9] * 255))
        cv2.drawContours(self.vis_image, [agent_arrow], 0, color, -1)

        return exploratory_node,adjacent_node,far_node,skeleton_mask,joint_node,voronoi_graph

    def best_node(self,my_node, exploratory_node, skeleton_mask, joint_node,planner_inputs,goal_id,voronoi_graph):
        ##对每个joint_node，求出my_node到joint_node的最短路径mask
        largest_keys=list(exploratory_node.keys())
        args=self.args
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = \
            planner_inputs['pose_pred']
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        r, c = start_y, start_x
        start = [int(r * 100.0 / args.map_resolution - gx1),
                 int(c * 100.0 / args.map_resolution - gy1)]
        start = pu.threshold_poses(start, skeleton_mask.shape)

        path=[]
        path_mask=[]
        rev_skeleton=skeleton_mask.copy()
        rev_skeleton=np.where(rev_skeleton==1,0,1)

        # 求出本地节点的相邻节点
        raw_my_node_neighbors=voronoi_graph[tuple(my_node)]
        iteration_node=voronoi_graph.copy()
        for i in range(len(iteration_node.keys())):
            iteration_node[list(iteration_node.keys())[i]]=0
        iteration_node[tuple(my_node)]=1

        my_node_neighbors= filter_node(raw_my_node_neighbors,my_node,voronoi_graph,iteration_node)


        neighbor_mask=np.zeros_like(skeleton_mask)
        for i in range(len(my_node_neighbors)):
            neighbor_mask[my_node_neighbors[i][0]-1:my_node_neighbors[i][0]+2,my_node_neighbors[i][1]-1:my_node_neighbors[i][1]+2]=1
        for i in range(len(largest_keys)):
            indices,weight = route_through_array(rev_skeleton, my_node, largest_keys[i],fully_connected=True)
            for j in range(len(indices)):
                if neighbor_mask[indices[j][0],indices[j][1]]==1:
                    indices=indices[j:]
                    break
            path.append(indices)
            indices = np.stack(indices, axis=-1)
            tmp_path_mask=np.zeros_like(skeleton_mask)
            tmp_path_mask[indices[0],indices[1]]=1
            path_mask.append(tmp_path_mask)

        # 把他们两两求并，抹除掉被包含的路径
        i=0
        kernel=np.ones((3,3),np.uint8)
        while i !=len(path_mask):
            j=i+1
            while j !=len(path_mask):
                dilation_i=cv2.dilate(path_mask[i].astype(np.uint8),kernel)
                dilation_j=cv2.dilate(path_mask[j].astype(np.uint8),kernel)
                if dilation_j[path[i][-1]]==1:
                    del path_mask[i]
                    del path[i]
                    break
                elif dilation_i[path[j][-1]]==1:
                    del path_mask[j]
                    del path[j]
                else:
                    j+=1
            i+=1

        # 求出这些路径的下一个节点
        path_next_node = [[] for i in range(len(path))]
        for i in range(len(path)):
            for j in range(len(my_node_neighbors)):
                if cv2.dilate(path_mask[i].astype(np.uint8),kernel)[my_node_neighbors[j][0],my_node_neighbors[j][1]]:
                    path_next_node[i]=my_node_neighbors[j]
                    break

        self.path_mask=path_mask
        # 对这些路径进行语义描述
        path_semantic=[[] for i in range(len(path))]
        kernel_semantic = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(30, 30))

        sem_map=planner_inputs['sem_map_pred'].copy()
        sem_dilate_list=[]
        for i in range(0,15):#一共有15个语义
            sem_dilate_list.append(cv2.dilate(np.where(sem_map==i,1,0).astype(np.uint8),kernel_semantic))
        path_description=[{} for i in range(len(path_semantic))]
        for i in range(len(path)):
            for j in range(len(sem_dilate_list)):
                if np.sum(np.logical_and(sem_dilate_list[j],path_mask[i]))>5:
                    path_semantic[i].append(coco_categories_inv[j])
                    tmp_label,tmp_num = measure.label(sem_dilate_list[j],connectivity=2,return_num=True)
                    for k in range(tmp_num):
                        indices = np.argwhere(tmp_label == k)
                        center = np.mean(indices, axis=0)
                        if (start[0]-center[0])**2+(start[1]-center[1])**2 < 80**2:
                            path_description[i][(int(center[0]),int(center[1]))]=coco_categories_inv[j]
                            break
        node_description=["" for i in range(len(my_node_neighbors))]
        for i in range(len(path_next_node)):
            if path_next_node[i]:
                for j in range(len(my_node_neighbors)):
                    if (path_next_node[i]==my_node_neighbors[j]) and list(path_description[i].keys())!=[]:
                        for k in range(len(list(path_description[i].keys()))):
                            if str(list(path_description[i].keys())[k]) not in node_description[j]:
                                node_description[j]+=path_description[i][list(path_description[i].keys())[k]]+" "+str(list(path_description[i].keys())[k])

        print(node_description)
        path_prompt="You are given areas' objects and coordinates, you should guess every zone's room type and describe them like \"A {room type} with {objects}\", if there is no objects, describe them like \"An empty room\""
        for i in range(len(node_description)):
            if node_description[i]!="":
                path_prompt+=" Zone "+str(i+1)+" : "+node_description[i]+"; "
            else:
                path_prompt+=" Zone "+str(i+1)+" : None ; "
        path_prompt+= ".reply with JSON form :{{\"zone_number\": $zone_number,\"description\": $description}"
        new_node_description = ["" for i in range(len(node_description))]
        error_count = 0
        while True:
            try:
                resp = client.chat.completions.create(
                    model="gpt-3.5-turbo-1106",
                    messages=[
                        {"role": "user", "content": path_prompt}
                    ],
                    temperature=0.8,
                )
                answer=resp.choices[0].message.content
                print(answer)
                # answer=answer.replace(" ","")
                answer=answer.replace("\n","")
                for i in range(len(new_node_description)):
                    left_bracket = answer.find('{')
                    index1=answer.find("description\":",left_bracket)
                    index2=answer.find("}",left_bracket)
                    new_node_description[i]=answer[index1+13:index2]
                    answer=answer[index2+1:]
                break
            except:
                error_count+=1
                time.sleep(1)
                if error_count >= 12:
                    print("Failed 12 times(path), stopping")
                    break

        my_node_neighbors_scores=[0 for i in range(len(my_node_neighbors))]
        # 预处理 caption 和 node
        my_node_neighbors_caption=[[] for i in range(len(my_node_neighbors))]

        nearest_idx=0
        nearest_dis=100000000
        for j in range(len(self.caption_locations)):
            if (start[0]+gx1-self.caption_locations[j][0])**2+(start[1]+gy1-self.caption_locations[j][1])**2<nearest_dis:
                nearest_idx=j
                nearest_dis=(start[0]+gx1-self.caption_locations[j][0])**2+(start[1]+gy1-self.caption_locations[j][1])**2

        for i in range(len(my_node_neighbors)):
            tmp_caption_location=[self.caption_locations[nearest_idx][0]-gx1,self.caption_locations[nearest_idx][1]-gy1]
            tmp_caption_location=pu.threshold_poses(tmp_caption_location, skeleton_mask.shape)
            tmp_orient=np.arctan2(my_node_neighbors[i][0]-tmp_caption_location[0],my_node_neighbors[i][1]-tmp_caption_location[1])*180/np.pi
            pair_orientation=round(tmp_orient/30)*30 if round(tmp_orient/30)*30!=180 else -180
            my_node_neighbors_caption[i]=self.orientation_caption[self.caption_locations[nearest_idx][0],self.caption_locations[nearest_idx][1]][pair_orientation]


        my_prompt="you have areas to go:"
        for i in range(len(my_node_neighbors_caption)):
            if "empty" in new_node_description[i]:
                my_prompt+="Areas "+str(i+1)+" : "+my_node_neighbors_caption[i]+";"
            else:
                my_prompt+="Areas "+str(i+1)+" : "+my_node_neighbors_caption[i]+", "+"and you may pass by "+new_node_description[i]+"; "

        my_prompt+=" Please estimate the probability of "+coco_categories_inv[goal_id]+" appearing at the these areas for me in [0,1].reply with JSON form :"+\
                    "{\{\"area_number\": $area_number,\"probability\": $probability}}"
        print(my_prompt)
        error_count = 0
        while True:
            try:
                resp = client.chat.completions.create(
                    model="gpt-3.5-turbo-1106",
                    messages=[
                        {'role': 'system', 'content': "You are performing a task of finding targets in an indoor environment, and you need to provide spatial logic and judgment based on common sense"},
                        {"role": "user", "content":my_prompt},
                    ],
                    temperature=0.8,
                )
                answer=resp.choices[0].message.content

                print(answer)
                answer=answer.replace(" ","")
                answer=answer.replace("\n","")
                for i in range(len(my_node_neighbors_caption)):
                    left_bracket = answer.find('{')
                    index1=answer.find("probability\":",left_bracket)
                    index2=answer.find("}",left_bracket)
                    probability=float(answer[index1+13:index2])
                    my_node_neighbors_scores[i]=probability
                    answer=answer[index2+1:]
                break
            except:
                error_count+=1
                time.sleep(1)
                if error_count >= 12:
                    print("Failed 12 times, stopping")
                    break

        # 开始收集数据
        os.makedirs("collected_data/file"+str(self.filename), exist_ok=True)
        my_prompt_1="you have areas to go:"
        for i in range(len(my_node_neighbors_caption)):
            if "empty" in new_node_description[i]:
                my_prompt_1+="Areas "+str(i+1)+" : "+my_node_neighbors_caption[i]+";"
            else:
                my_prompt_1+="Areas "+str(i+1)+" : "+my_node_neighbors_caption[i]+", "+"and you may pass by "+new_node_description[i]+"; "

        my_prompt_1+=" Please estimate the probability of "+coco_categories_inv[goal_id]+" appearing at the these areas for me in [0,1] and explain why .reply with JSON form :"+\
                    "{\{\"area_number\": $area_number,\"probability\": $probability,\"reason\": $reason}}"
        error_count = 0
        while True:
            try:
                resp = client.chat.completions.create(
                    model="gpt-3.5-turbo-1106",
                    messages=[
                        {'role': 'system', 'content': "You are performing a task of finding targets in an indoor environment, and you need to provide spatial logic and judgment based on common sense"},
                        {"role": "user", "content":my_prompt_1},
                    ],
                    temperature=0.8,
                )
                answer_1=resp.choices[0].message.content
                break
            except:
                error_count+=1
                time.sleep(1)
                if error_count >= 12:
                    print("Failed 12 times, stopping")
                    break  
    
        file_path = os.path.join("collected_data/file"+str(self.filename), "1.txt")
        with open(file_path, "w") as file:
            file.write(my_prompt_1)
        file_path = os.path.join("collected_data/file"+str(self.filename), "2.txt")
        with open(file_path, "w") as file:
            file.write(answer_1)

        # 开始收集数据 2
        my_prompt_1="you have areas to go:"
        for i in range(len(my_node_neighbors_caption)):
            if "empty" in new_node_description[i]:
                my_prompt_1+="Areas "+str(i+1)+" : "+my_node_neighbors_caption[i]+";"
            else:
                my_prompt_1+="Areas "+str(i+1)+" : "+my_node_neighbors_caption[i]+"; "

        my_prompt_1+=" Please estimate the probability of "+coco_categories_inv[goal_id]+" appearing at the these areas for me in [0,1]. Please analyze in details and explain why .Reply with JSON form :"+\
                    "{\{\"area_number\": $area_number,\"probability\": $probability,\"reason\": $reason}}"
        error_count = 0
        while True:
            try:
                resp = client.chat.completions.create(
                    model="gpt-3.5-turbo-1106",
                    messages=[
                        {'role': 'system', 'content': "You are performing a task of finding targets in an indoor environment, and you need to provide spatial logic and judgment based on common sense"},
                        {"role": "user", "content":my_prompt_1},
                    ],
                    temperature=0.8,
                )
                answer_1=resp.choices[0].message.content
                break
            except:
                error_count+=1
                time.sleep(1)
                if error_count >= 12:
                    print("Failed 12 times, stopping")
                    break  
    
        file_path = os.path.join("collected_data/file"+str(self.filename), "3.txt")
        with open(file_path, "w") as file:
            file.write(my_prompt_1)
        file_path = os.path.join("collected_data/file"+str(self.filename), "4.txt")
        with open(file_path, "w") as file:
            file.write(answer_1)

        # 开始收集数据 3
        my_prompt_1="you have areas to go:"
        for i in range(len(my_node_neighbors_caption)):
            if "empty" in new_node_description[i]:
                my_prompt_1+="Areas "+str(i+1)+" : It's empty ;"
            else:
                my_prompt_1+="Areas "+str(i+1)+" : You will pass by "+new_node_description[i]+"; "
        my_prompt_1+=" Please estimate the probability of "+coco_categories_inv[goal_id]+" appearing at the these areas for me in [0,1]. Please analyze in details and explain why .Reply with JSON form :"+\
                    "{\{\"area_number\": $area_number,\"probability\": $probability,\"reason\": $reason}}"
        error_count = 0
        while True:
            try:
                resp = client.chat.completions.create(
                    model="gpt-3.5-turbo-1106",
                    messages=[
                        {'role': 'system', 'content': "You are performing a task of finding targets in an indoor environment, and you need to provide spatial logic and judgment based on common sense"},
                        {"role": "user", "content":my_prompt_1},
                    ],
                    temperature=0.8,
                )
                answer_1=resp.choices[0].message.content
                break
            except:
                error_count+=1
                time.sleep(1)
                if error_count >= 12:
                    print("Failed 12 times, stopping")
                    break  
    
        file_path = os.path.join("collected_data/file"+str(self.filename), "5.txt")
        with open(file_path, "w") as file:
            file.write(my_prompt_1)
        file_path = os.path.join("collected_data/file"+str(self.filename), "6.txt")
        with open(file_path, "w") as file:
            file.write(answer_1)    
            
        cv2.imwrite("collected_data/file"+str(self.filename)+"/image.png", self.vis_image)
        self.filename=self.filename+1

        for i in range(len(my_node_neighbors)):
            for j in range(len(path_semantic)):
                if path_next_node[j]==my_node_neighbors[i]:
                    my_node_neighbors_scores[i]+=2
                    break
            if self.joint_visited[gx1:gx2, gy1:gy2][my_node_neighbors[i][0],my_node_neighbors[i][1]]==0:
                my_node_neighbors_scores[i]+=1

        return my_node_neighbors[np.argmax(my_node_neighbors_scores)]

def area_mask(arr, x, y, r):
    """生成一个圆形的mask"""
    h, w = arr.shape
    yy, xx = np.meshgrid(np.arange(h), np.arange(w))
    dist = (xx - x) ** 2 + (yy - y) ** 2
    mask = dist <= r**2
    return mask.astype(np.int)

def search_adjacent(start,points,local_ob_map):
    """寻找最近的节点"""
    selem = skimage.morphology.disk(1)
    traversible = skimage.morphology.binary_dilation(local_ob_map, selem) != True
    planner = FMMPlanner(traversible)
    goal_pose_map = np.zeros((local_ob_map.shape))
    goal_pose_map[int(start[0]), int(start[1])] = 1
    planner.set_multi_goal(goal_pose_map)
    min_dis=10000000
    min_point=[]
    my_dict={}

    for point in points:
        dist = planner.fmm_dist[int(point[0]), int(point[1])]
        if dist<min_dis:
            min_dis=dist
            min_point=point
        my_dict[tuple(point)]=dist

    return {tuple(min_point):min_dis},my_dict

def search_far(start,points):
    """寻找最远的节点"""
    max_dis=0
    max_point=[]
    my_dict={}
    for point in points:
        dis=(point[0]-start[0])**2+(point[1]-start[1])**2
        if dis>max_dis:
            max_dis=dis
            max_point=point
        my_dict[tuple(point)]=dis
    return {tuple(max_point):max_dis},my_dict

def build_graph(matrix, points,vis_mask,true_exp_mask,map_mask):
    """构建点之间的连通性"""
    m, n = matrix.shape
    graph = {tuple(point): [] for point in points}
    edges = {tuple(point): [] for point in points}#节点之间的路径长度
    abs_edges = {tuple(point): [] for point in points}#节点之间的绝对距离
    points = [tuple(point) for point in points]
    visited = np.zeros_like(matrix, dtype=bool)

    distance = np.full(matrix.shape,np.inf, dtype=int)#路径长度
    abs_distance = np.full(matrix.shape, 0 , dtype=int)#绝对距离

    directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (-1,-1),(1,1),(-1,1),(1,-1)]
    points_unexplore_area={tuple(point): 0 for point in points}

    for start in points:
        visited = np.zeros_like(matrix, dtype=bool)
        distance = np.full(matrix.shape,np.inf, dtype=int)
        queue = Queue()
        queue.put(start)
        visited[start[0],start[1]] = True
        distance[start[0],start[1]]=-1
        while not queue.empty():
            x, y = queue.get()
            flag=0
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (0 <= nx < m) and (0 <= ny < n) and ((nx,ny) in points) and not visited[nx, ny] and ((nx,ny)!=start):
                    graph[start].append((nx, ny))
                    edges[start].append(distance[x,y]+1)
                    abs_edges[start].append(abs_distance[x,y])
                    # graph[(nx, ny)].append(start)
                    flag=1
            if flag==1:
                continue
            for dx, dy in directions:
                nx, ny = x + dx, y + dy

                if 0 <= nx < m and 0 <= ny < n and matrix[nx, ny] == 1 and not visited[nx, ny]:
                    distance[nx,ny]=distance[x,y]+1
                    visited[nx, ny] = True
                    queue.put((nx, ny))
                    if vis_mask[nx,ny]==1:
                        abs_distance[nx,ny]=abs_distance[x,y]-1
                    else:
                        abs_distance[nx,ny]=abs_distance[x,y]+1
    return graph

def filter_node(raw_my_node_neighbors,my_node,voronoi_graph,iteration_node):
    new_node_neighbor=[]

    for i in range(len(raw_my_node_neighbors)):
        if (raw_my_node_neighbors[i][0]-my_node[0])**2+(raw_my_node_neighbors[i][1]-my_node[1])**2>15**2:
            if iteration_node[raw_my_node_neighbors[i]]!=1:
                new_node_neighbor.append(raw_my_node_neighbors[i])
            iteration_node[raw_my_node_neighbors[i]]=1
        else:
            if iteration_node[raw_my_node_neighbors[i]]==1:
                continue
            if len(voronoi_graph[tuple(raw_my_node_neighbors[i])])==1:
                new_node_neighbor.append(raw_my_node_neighbors[i])
                iteration_node[raw_my_node_neighbors[i]]=1
                continue
            iteration_node[raw_my_node_neighbors[i]]=1
            new_node_neighbor+=filter_node(voronoi_graph[tuple(raw_my_node_neighbors[i])],my_node,voronoi_graph,iteration_node)

    return new_node_neighbor


