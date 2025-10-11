from collections import deque, defaultdict
import os
import logging
import time
import json
import gym
import torch.nn as nn
import torch
import numpy as np
from model import  Semantic_Mapping
from utils.storage import GlobalRolloutStorage
from envs import make_vec_envs
from arguments import get_args
import envs.utils.pose as pu
from queue import Queue
from model import preprocess_exp_map
import cv2
from skimage import measure
import skimage.morphology
from envs.utils.fmm_planner import FMMPlanner
os.environ["OMP_NUM_THREADS"] = "1"


def main(epi):

    torch.cuda.empty_cache()
    args = get_args()
    clr_flag=0
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Setup Logging
    log_dir = "{}/models/{}/".format(args.dump_location, args.exp_name)
    dump_dir = "{}/dump/{}/".format(args.dump_location, args.exp_name)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    logging.basicConfig(
        filename=log_dir + 'train.log',
        level=logging.INFO)
    print("Dumping at {}".format(log_dir))
    print(args)
    logging.info(args)

    # Logging and loss variables
    num_scenes = args.num_processes
    num_episodes = int(args.num_eval_episodes)
    device = args.device = torch.device("cpu" if args.cuda else "cpu")
    g_masks = torch.ones(num_scenes).float().to(device)

    best_g_reward = -np.inf

    if args.eval:
        episode_success = []
        episode_spl = []
        episode_dist = []
        episode_sca=[]
        episode_sea=[]
        for _ in range(args.num_processes):
            episode_success.append(deque(maxlen=num_episodes))
            episode_spl.append(deque(maxlen=num_episodes))
            episode_dist.append(deque(maxlen=num_episodes))
            episode_sca.append(deque(maxlen=num_episodes))
            episode_sea.append(deque(maxlen=num_episodes))


    else:
        episode_success = deque(maxlen=1000)
        episode_spl = deque(maxlen=1000)
        episode_dist = deque(maxlen=1000)
        episode_sca = deque(maxlen=1000)
        episode_sea = deque(maxlen=1000)

    finished = np.zeros((args.num_processes))
    wait_env = np.zeros((args.num_processes))

    g_episode_rewards = deque(maxlen=1000)

    g_value_losses = deque(maxlen=1000)
    g_action_losses = deque(maxlen=1000)
    g_dist_entropies = deque(maxlen=1000)

    per_step_g_rewards = deque(maxlen=1000)

    g_process_rewards = np.zeros((num_scenes))
    stair_flag = np.zeros((num_scenes))
    clear_flag = np.zeros((num_scenes))
    # Starting environments
    torch.set_num_threads(1)
    envs = make_vec_envs(args)
    obs, infos = envs.reset()

    torch.set_grad_enabled(False)

    # Initialize map variables:
    # Full map consists of multiple channels containing the following:
    # 1. Obstacle Map
    # 2. Exploread Area
    # 3. Current Agent Location
    # 4. Past Agent Locations
    # 5,6,7,.. : Semantic Categories
    nc = args.num_sem_categories + 4  # num channels

    # Calculating full and local map sizes
    map_size = args.map_size_cm // args.map_resolution
    full_w, full_h = map_size, map_size
    local_w = int(full_w / args.global_downscaling)
    local_h = int(full_h / args.global_downscaling)

    # Initializing full and local map
    full_map = torch.zeros(num_scenes, nc, full_w, full_h).float().to(device)
    local_map = torch.zeros(num_scenes, nc, local_w,
                            local_h).float().to(device)
    local_ob_map = np.zeros((num_scenes, local_w,
                            local_h))

    local_ex_map = np.zeros((num_scenes, local_w,
                            local_h))
    target_edge_map = np.zeros((num_scenes, local_w,
                            local_h))
    target_point_map = np.zeros((num_scenes, local_w,
                            local_h))
    # Initial full and local pose
    full_pose = torch.zeros(num_scenes, 3).float().to(device)
    local_pose = torch.zeros(num_scenes, 3).float().to(device)

    # Origin of local map
    origins = np.zeros((num_scenes, 3))

    # Local Map Boundaries
    lmb = np.zeros((num_scenes, 4)).astype(int)

    # Planner pose inputs has 7 dimensions
    # 1-3 store continuous global agent location
    # 4-7 store local map boundaries
    planner_pose_inputs = np.zeros((num_scenes, 7))

    def get_local_map_boundaries(agent_loc, local_sizes, full_sizes):
        loc_r, loc_c = agent_loc
        local_w, local_h = local_sizes
        full_w, full_h = full_sizes

        if args.global_downscaling > 1:
            gx1, gy1 = loc_r - local_w // 2, loc_c - local_h // 2
            gx2, gy2 = gx1 + local_w, gy1 + local_h
            if gx1 < 0:
                gx1, gx2 = 0, local_w
            if gx2 > full_w:
                gx1, gx2 = full_w - local_w, full_w

            if gy1 < 0:
                gy1, gy2 = 0, local_h
            if gy2 > full_h:
                gy1, gy2 = full_h - local_h, full_h
        else:
            gx1, gx2, gy1, gy2 = 0, full_w, 0, full_h

        return [gx1, gx2, gy1, gy2]

    def init_map_and_pose():
        full_map.fill_(0.)
        full_pose.fill_(0.)
        full_pose[:, :2] = args.map_size_cm / 100.0 / 2.0

        locs = full_pose.cpu().numpy()
        planner_pose_inputs[:, :3] = locs
        for e in range(num_scenes):
            r, c = locs[e, 1], locs[e, 0]
            loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                            int(c * 100.0 / args.map_resolution)]

            full_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0

            lmb[e] = get_local_map_boundaries((loc_r, loc_c),
                                              (local_w, local_h),
                                              (full_w, full_h))

            planner_pose_inputs[e, 3:] = lmb[e]
            origins[e] = [lmb[e][2] * args.map_resolution / 100.0,
                          lmb[e][0] * args.map_resolution / 100.0, 0.]

        for e in range(num_scenes):
            local_map[e] = full_map[e, :,
                                    lmb[e, 0]:lmb[e, 1],
                                    lmb[e, 2]:lmb[e, 3]]
            local_pose[e] = full_pose[e] - \
                torch.from_numpy(origins[e]).to(device).float()

    def init_map_and_pose_for_env(e):
        full_map[e].fill_(0.)
        full_pose[e].fill_(0.)
        local_ob_map[e]=np.zeros((local_w,
                            local_h))
        local_ex_map[e]=np.zeros((local_w,
                            local_h))
        target_edge_map[e]=np.zeros((local_w,
                            local_h))
        target_point_map[e]=np.zeros((local_w,
                            local_h))
        stair_flag[e] = 0
        clear_flag[e] = 0




        full_pose[e, :2] = args.map_size_cm / 100.0 / 2.0
        stair_flag[e] = 0
        clear_flag[e] = 0
        locs = full_pose[e].cpu().numpy()
        planner_pose_inputs[e, :3] = locs
        r, c = locs[1], locs[0]
        loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                        int(c * 100.0 / args.map_resolution)]

        full_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0

        lmb[e] = get_local_map_boundaries((loc_r, loc_c),
                                          (local_w, local_h),
                                          (full_w, full_h))

        planner_pose_inputs[e, 3:] = lmb[e]
        origins[e] = [lmb[e][2] * args.map_resolution / 100.0,
                      lmb[e][0] * args.map_resolution / 100.0, 0.]

        local_map[e] = full_map[e, :, lmb[e, 0]:lmb[e, 1], lmb[e, 2]:lmb[e, 3]]
        local_pose[e] = full_pose[e] - \
            torch.from_numpy(origins[e]).to(device).float()



    init_map_and_pose()
    def remove_small_points(local_ob_map, image, threshold_point, pose):
        # print("goal_cat_id: ", goal_cat_id)
        # print("sem: ", sem.shape)
        selem = skimage.morphology.disk(1)
        traversible = skimage.morphology.binary_dilation(
            local_ob_map, selem) != True
        # traversible = 1 - traversible
        planner = FMMPlanner(traversible)
        goal_pose_map = np.zeros((local_ob_map.shape))
        pose_x = int(pose[0].cpu()) if int(pose[0].cpu()) < local_w-1 else local_w-1
        pose_y = int(pose[1].cpu()) if int(pose[1].cpu()) < local_w-1 else local_w-1
        goal_pose_map[pose_x, pose_y] = 1
        # goal_map = skimage.morphology.binary_dilation(
        #     goal_pose_map, selem) != True
        # goal_map = 1 - goal_map
        planner.set_multi_goal(goal_pose_map)

        img_label, num = measure.label(image, connectivity=2, return_num=True)#输出二值图像中所有的连通域
        props = measure.regionprops(img_label)#输出连通域的属性，包括面积等
        # print("img_label.shape: ", img_label.shape) # 480*480
        # print("img_label.dtype: ", img_label.dtype) # 480*480
        Goal_edge = np.zeros((img_label.shape[0], img_label.shape[1]))
        Goal_point = np.zeros(img_label.shape)
        Goal_score = []

        dict_cost = {}
        for i in range(1, len(props)):
            # print("area: ", props[i].area)
            # dist = pu.get_l2_distance(props[i].centroid[0], pose[0], props[i].centroid[1], pose[1])
            dist = planner.fmm_dist[int(props[i].centroid[0]), int(props[i].centroid[1])] * 5
            dist_s = 8 if dist < 300 else 0
            
            cost = props[i].area + dist_s

            if props[i].area > threshold_point and dist > 150 and dist < 500:
                dict_cost[i] = cost
        
        if dict_cost:
            dict_cost = sorted(dict_cost.items(), key=lambda x: x[1], reverse=True)
            
            # print(dict_cost)
            for i, (key, value) in enumerate(dict_cost):
                # print(i, key)
                Goal_edge[img_label == key + 1] = 1
                Goal_point[int(props[key].centroid[0]), int(props[key].centroid[1])] = i+1 #
                Goal_score.append(value)
                if i == 3:
                    break

        return Goal_edge, Goal_point, Goal_score

    # Global policy observation space
    ngc = 8 + args.num_sem_categories


    # Semantic Mapping
    sem_map_module = Semantic_Mapping(args).to(device)
    sem_map_module.eval()

    # Global policy


    # Predict semantic map from frame 1
    poses = torch.from_numpy(np.asarray(
        [infos[env_idx]['sensor_pose'] for env_idx in range(num_scenes)])
    ).float().to(device)

    eve_angle = np.asarray(
        [infos[env_idx]['eve_angle'] for env_idx
            in range(num_scenes)])
    

    _, local_map, local_map_stair, local_pose = \
        sem_map_module(obs, poses, local_map, local_pose, eve_angle)


    # Compute Global policy input
    locs = local_pose.cpu().numpy()

    for e in range(num_scenes):
        r, c = locs[e, 1], locs[e, 0]
        loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                        int(c * 100.0 / args.map_resolution)]

        local_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.



    goal_cat_id = torch.from_numpy(np.asarray(
        [infos[env_idx]['goal_cat_id'] for env_idx
         in range(num_scenes)]))




    # Run Global Policy (global_goals = Long-Term Goal)


    goal_maps = [np.zeros((local_w, local_h)) for _ in range(num_scenes)]



    planner_inputs = [{} for e in range(num_scenes)]
    for e, p_input in enumerate(planner_inputs):
        p_input['map_pred'] = local_map[e, 0, :, :].cpu().numpy()
        p_input['exp_pred'] = local_map[e, 1, :, :].cpu().numpy()
        p_input['pose_pred'] = planner_pose_inputs[e]
        p_input['goal'] = goal_maps[e]  # global_goals[e]
        p_input['new_goal'] = 1
        p_input['found_goal'] = 0
        p_input['wait'] = wait_env[e] or finished[e]
        p_input['clr']=0
        if args.visualize or args.print_images:
            p_input['map_edge'] = target_edge_map[e]
            local_map[e, -1, :, :] = 1e-5
            p_input['sem_map_pred'] = local_map[e, 4:, :, :
                                                ].argmax(0).cpu().numpy()

    obs, _, done, infos = envs.plan_act_and_preprocess(planner_inputs)

    start = time.time()
    g_reward = 0

    torch.set_grad_enabled(False)
    spl_per_category = defaultdict(list)
    success_per_category = defaultdict(list)

    for step in range(args.num_training_frames // args.num_processes + 1):
        if finished.sum() == args.num_processes:
            break

        g_step = (step // args.num_local_steps) % args.num_global_steps
        l_step = step % args.num_local_steps

        # ------------------------------------------------------------------
        # Reinitialize variables when episode ends
        l_masks = torch.FloatTensor([0 if x else 1
                                     for x in done]).to(device)
        g_masks *= l_masks

        for e, x in enumerate(done):
            if x:
                spl = infos[e]['spl']
                success = infos[e]['success']
                sca = infos[e]['sca']
                sea = infos[e]['sea']
                dist = infos[e]['distance_to_goal']
                spl_per_category[infos[e]['goal_name']].append(spl)
                success_per_category[infos[e]['goal_name']].append(success)
                if args.eval:
                    episode_success[e].append(success)
                    episode_spl[e].append(spl)
                    episode_dist[e].append(dist)
                    episode_sca[e].append(sca)
                    episode_sea[e].append(sea)
                    if len(episode_success[e]) == num_episodes:
                        finished[e] = 1
                else:
                    episode_success.append(success)
                    episode_spl.append(spl)
                    episode_dist.append(dist)
                    episode_sca[e].append(sca)
                    episode_sea[e].append(sea)
                wait_env[e] = 1.

                init_map_and_pose_for_env(e)
                clr_flag=1
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Semantic Mapping Module
        poses = torch.from_numpy(np.asarray(
            [infos[env_idx]['sensor_pose'] for env_idx
             in range(num_scenes)])
        ).float().to(device)

        eve_angle = np.asarray(
            [infos[env_idx]['eve_angle'] for env_idx
                in range(num_scenes)])
        _, local_map, local_map_stair, local_pose = \
            sem_map_module(obs, poses, local_map, local_pose,eve_angle)
        
        # local_map[:, 0, :, :][local_map[:, 18, :, :] > 0] = 0
        locs = local_pose.cpu().numpy()
        planner_pose_inputs[:, :3] = locs + origins
        local_map[:, 2, :, :].fill_(0.)  # Resetting current location channel
        for e in range(num_scenes):
            r, c = locs[e, 1], locs[e, 0]
            loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                            int(c * 100.0 / args.map_resolution)]
            local_map[e, 2:4, loc_r - 2:loc_r + 3, loc_c - 2:loc_c + 3] = 1.
            if args.eval:
            # # clear the obstacle during the stairs
                if loc_r > local_w: loc_r = local_w-1
                if loc_c > local_h: loc_c = local_h-1
                if local_map[e, 18, loc_r, loc_c] > 0.5:
                    stair_flag[e] = 1

                if stair_flag[e]:
                    # must > 0
                    if torch.any(local_map[e, 18, :, :] > 0.5):
                        local_map[e, 0, :, :] = local_map_stair[e, 0, :, :]
                    local_map[e, 0, :, :] = local_map_stair[e, 0, :, :]

        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Global Policy
        if l_step == args.num_local_steps - 1:
            # For every global step, update the full and local maps
            for e in range(num_scenes):
                if wait_env[e] == 1:  # New episode
                    wait_env[e] = 0.


                full_map[e, :, lmb[e, 0]:lmb[e, 1], lmb[e, 2]:lmb[e, 3]] = \
                    local_map[e]
                full_pose[e] = local_pose[e] + \
                    torch.from_numpy(origins[e]).to(device).float()

                locs = full_pose[e].cpu().numpy()
                r, c = locs[1], locs[0]
                loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                                int(c * 100.0 / args.map_resolution)]

                lmb[e] = get_local_map_boundaries((loc_r, loc_c),
                                                  (local_w, local_h),
                                                  (full_w, full_h))

                planner_pose_inputs[e, 3:] = lmb[e]
                origins[e] = [lmb[e][2] * args.map_resolution / 100.0,
                              lmb[e][0] * args.map_resolution / 100.0, 0.]

                local_map[e] = full_map[e, :,
                                        lmb[e, 0]:lmb[e, 1],
                                        lmb[e, 2]:lmb[e, 3]]
                local_pose[e] = full_pose[e] - \
                    torch.from_numpy(origins[e]).to(device).float()
        for e in range(num_scenes):
            if infos[e]['clear_flag']:
                clear_flag[e] = 1

            if clear_flag[e]:
                local_map[e].fill_(0.)
                clear_flag[e] = 0




        for e in range(num_scenes):
            _local_ob_map = local_map[e][0].cpu().numpy()
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
            local_ob_map[e] = cv2.dilate(_local_ob_map, kernel)
            show_ex = cv2.inRange(local_map[e][1].cpu().numpy(),0.1,1)
            kernel = np.ones((10, 10), dtype=np.uint8)
            free_map = cv2.morphologyEx(show_ex, cv2.MORPH_CLOSE, kernel)   
            contours,_=cv2.findContours(free_map, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
            if len(contours)>0:
                contour = max(contours, key = cv2.contourArea)
                cv2.drawContours(local_ex_map[e],contour,-1,1,1)
            local_ex_map[e, 0:2, 0:local_w]=0.0
            local_ex_map[e, local_w-2:local_w, 0:local_w-1]=0.0
            local_ex_map[e, 0:local_w, 0:2]=0.0
            local_ex_map[e, 0:local_w, local_w-2:local_w]=0.0
            target_edge = np.zeros((local_w, local_h))
            target_edge = local_ex_map[e]-local_ob_map[e]

            target_edge[target_edge>0.8]=1.0
            target_edge[target_edge!=1.0]=0.0
            local_pose_map = [local_pose[e][1]*100/args.map_resolution, local_pose[e][0]*100/args.map_resolution]
            target_edge_map[e], target_point_map[e], Goal_score = remove_small_points(_local_ob_map, target_edge, 4, local_pose_map) 
            local_ob_map[e]=np.zeros((local_w,
                        local_h))
            local_ex_map[e]=np.zeros((local_w,
                        local_h))






        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Update long-term goal if target object is found
        found_goal = [0 for _ in range(num_scenes)]
        goal_maps = [np.zeros((local_w, local_h)) for _ in range(num_scenes)]



        for e in range(num_scenes):
            cn = infos[e]['goal_cat_id'] + 4
            if local_map[e, cn, :, :].sum() != 0.:
                cat_semantic_map = local_map[e, cn, :, :].cpu().numpy()
                cat_semantic_scores = cat_semantic_map
                cat_semantic_scores[cat_semantic_scores > 0] = 1.
                goal_maps[e] = cat_semantic_scores
                found_goal[e] = 1
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Take action and get next observation
        planner_inputs = [{} for e in range(num_scenes)]
        #预处理exp_map
        # local_map[0,1,:,:] = torch.from_numpy(np.logical_or(local_map[0,1,:,:].cpu().numpy(),preprocess_exp_map(local_map[0,1,:,:].cpu().numpy(),local_map[e, 0, :, :].cpu().numpy(),planner_pose_inputs[0]))).to(device)
        for e, p_input in enumerate(planner_inputs):
            p_input['map_pred'] = local_map[e, 0, :, :].cpu().numpy()
            p_input['exp_pred'] = local_map[e, 1, :, :].cpu().numpy()
            p_input['pose_pred'] = planner_pose_inputs[e]
            p_input['goal'] = goal_maps[e]  # global_goals[e]
            p_input['new_goal'] = l_step == args.num_local_steps - 1
            p_input['found_goal'] = found_goal[e]
            p_input['wait'] = wait_env[e] or finished[e]
            if clr_flag==1:
                p_input['clr']=1
                clr_flag=0
            else:
                p_input['clr']=0
            if args.visualize or args.print_images:
                p_input['map_edge'] = target_edge_map[e]
                local_map[e, -1, :, :] = 1e-5
                p_input['sem_map_pred'] = local_map[e, 4:, :,
                                                    :].argmax(0).cpu().numpy()
                


        obs, _, done, infos = envs.plan_act_and_preprocess(planner_inputs)
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Training

        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Logging
        if step % args.log_interval == 0:
            end = time.time()
            time_elapsed = time.gmtime(end - start)
            log = " ".join([
                "Time: {0:0=2d}d".format(time_elapsed.tm_mday - 1),
                "{},".format(time.strftime("%Hh %Mm %Ss", time_elapsed)),
                "num timesteps {},".format(step * num_scenes),
                "FPS {},".format(int(step * num_scenes / (end - start)))
            ])

            log += "\n\tRewards:"

            if len(g_episode_rewards) > 0:
                log += " ".join([
                    " Global step mean/med rew:",
                    "{:.4f}/{:.4f},".format(
                        np.mean(per_step_g_rewards),
                        np.median(per_step_g_rewards)),
                    " Global eps mean/med/min/max eps rew:",
                    "{:.3f}/{:.3f}/{:.3f}/{:.3f},".format(
                        np.mean(g_episode_rewards),
                        np.median(g_episode_rewards),
                        np.min(g_episode_rewards),
                        np.max(g_episode_rewards))
                ])

            if args.eval:
                total_success = []
                total_spl = []
                total_dist = []
                total_sca = []
                total_sea = []
                for e in range(args.num_processes):
                    for acc in episode_success[e]:
                        total_success.append(acc)
                    for dist in episode_dist[e]:
                        total_dist.append(dist)
                    for spl in episode_spl[e]:
                        total_spl.append(spl)
                    for sca in episode_sca[e]:
                        total_sca.append(spl)
                    for sea in episode_sea[e]:
                        total_sea.append(spl)

                if len(total_spl) > 0:
                    log+="ObjectNav succ/spl/dtg:"
                    log += " {:.3f}/{:.3f}/{:.3f}/{:.3f} ({:.0f}),".format(
                        np.mean(total_success),
                        np.mean(total_spl),
                        np.mean(total_sca),
                        np.mean(total_sea),
                        len(total_spl),)
                    
                    
                    

            log += "\nsuccess and failure reasons counts\n"
            
            log += "Start number: {} | Success counts:{} | Exploration fail: {} | Collision fail: {} | Detection fail: {} \n".format(infos[0]['episode_id'],
                                          infos[0]['fail_case']['success'],
                                          infos[0]['fail_case']['exploration'],
                                          infos[0]['fail_case']['collision'],
                                          infos[0]['fail_case']['detection'])            



            note=open('gt.txt',mode='a')
            note.write(log)
            note.close
            print(log)
            logging.info(log)


        # ------------------------------------------------------------------

    # Print and save model performance numbers during evaluation
    if args.eval:
        print("Dumping eval details...")
        
        total_success = []
        total_spl = []
        total_dist = []
        for e in range(args.num_processes):
            for acc in episode_success[e]:
                total_success.append(acc)
            for dist in episode_dist[e]:
                total_dist.append(dist)
            for spl in episode_spl[e]:
                total_spl.append(spl)

        if len(total_spl) > 0:
            log+="ObjectNav succ/spl/dtg:"
            log += " {:.3f}/{:.3f}/{:.3f}/{:.3f} ({:.0f}),".format(
                np.mean(total_success),
                np.mean(total_spl),
                np.mean(total_sca),
                np.mean(total_sea),
                len(total_spl),)
        log += "\nsuccess and failure reasons counts\n"

        log += "Start number: {} | Success counts:{} | Exploration fail: {} | Collision fail: {} | Detection fail: {} \n".format(infos[0]['episode_id'],
                                      infos[0]['fail_case']['success'],
                                      infos[0]['fail_case']['exploration'],
                                      infos[0]['fail_case']['collision'],
                                      infos[0]['fail_case']['detection'])      

        note=open('gt.txt',mode='a')
        note.write(log)
        note.close
        print(log)
        logging.info(log)

        with open('{}/{}_spl_per_cat_pred_thr.json'.format(
                dump_dir, args.split), 'w') as f:
            json.dump(spl_per_category, f)

        with open('{}/{}_success_per_cat_pred_thr.json'.format(
                dump_dir, args.split), 'w') as f:
            json.dump(success_per_category, f)


if __name__ == "__main__":

    main(0)
