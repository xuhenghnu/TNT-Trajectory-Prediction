# About: script to processing argoverse forecasting dataset
# Author: Jianbang LIU @ RPAI, CUHK
# Date: 2021.07.16

import os
import argparse
from os.path import join
import copy
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy import sparse
from pathlib import Path

import warnings

# import torch
from torch.utils.data import Dataset, DataLoader

from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.visualization.visualize_sequences import viz_sequence
from argoverse.utils.mpl_plotting_utils import visualize_centerline


from core.util.cubic_spline import Spline2D

# 当前文件路径
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent.parent
print('project_root: ', project_root)

warnings.filterwarnings("ignore")

RESCALE_LENGTH = 1.0    # the rescale length th turn the lane vector into equal distance pieces


class ArgoversePreprocessor(Dataset):
    def __init__(self,
                 param_dict,
                 split="train"):
        
        self.raw_dir = param_dict['raw_dir']
        self.save_dir = param_dict['save_dir']
        self.obs_horizon = param_dict['obs_horizon']
        self.obs_range = param_dict['obs_range']
        self.pred_horizon = param_dict['pred_horizon']
        self.normalized = param_dict['normalized']
        self.map_feat = True

        self.LANE_WIDTH = {'MIA': 3.84, 'PIT': 3.97}
        self.COLOR_DICT = {"AGENT": "#d33e4c", "OTHERS": "#d3e8ef", "AV": "#007672"}

        self.split = split

        # Argoverse数据集的地图
        self.am = ArgoverseMap()
        # ArgoverseForecasting数据集的api接口
        self.afl = ArgoverseForecastingLoader(os.path.join(self.raw_dir, self.split+"_obs" if split == "test" else split))

    # DataSet必须重写__len__和__getitem__
    def __len__(self):
        return len(self.afl)

    def __getitem__(self, idx):
        # 获取数据集中的第idx条数据路径
        # API： 在ArgoverseForecastingLoader初始化时会将所有的csv文件路径存储在seq_list中
        f_path = self.afl.seq_list[idx] 
        # 获取数据集中的第idx条数据
        # API： 返回的是ArgoverseForecastingLoader对象，并将该对象的current_seq设置为第idx条数据
        seq = self.afl.get(f_path)
        # 数据做一个拷贝到df
        # API： seq_df返回的是通过Pandas读取的csv文件内容，数据结构是DataFrame
        df = copy.deepcopy(seq.seq_df)
        # 拆分出路径，文件名（就是seq_id），后缀名
        path, seq_f_name_ext = os.path.split(f_path)
        seq_f_name, ext = os.path.splitext(seq_f_name_ext)
        # 处理数据，也就是训练用的数据
        df_processed = self.process(df, seq_f_name, self.map_feat)
        # 保存处理后的数据，也就是训练用的数据
        self.save(df_processed, seq_f_name, self.save_dir)
        return []

    def process(self, dataframe: pd.DataFrame, seq_id, map_feat=True):
        # 提取AGENT和其他目标的轨迹，以及数据所在城市（没有障碍物类型，这个数据集没有区分类型）
        data = self.read_argo_data(dataframe)
        # 提取目标的其他数据
        data = self.get_obj_feats(data)
        # 提取地图属性（主要是车道属性）
        data['graph'] = self.get_lane_graph(data)
        data['seq_id'] = seq_id
        
        '''
        data = {
            'trajs': [traj1, traj2, traj3, ...],
            'steps': [step1, step2, step3, ...],
            'city': 'MIA',
            'orig': 坐标转换的原点（地图坐标系下）
            'theta': 坐标转换的角度（地图坐标系下）
            'rot': 旋转矩阵
            'feats': 所有目标坐标转换后的观测轨迹 Nx3(最后一列是1.0,有点像置信度)
            'has_obss': 标记所有目标的这些时间步的观测轨迹点存在
            'gt_preds': 所有目标坐标转换后的预测轨迹真值 Nx2
            'has_preds': 标记所有目标的这些时间步的预测轨迹点存在
            'tar_candts': 候选点
            'gt_candts': 候选点真值
            'gt_tar_offset': 候选点偏移真值
            'ref_ctr_lines': 所有中心线的拟合曲线
            'ref_cetr_idx': 预测轨迹所在的车道线的索引
            'graph': {
              "ctrs": 所有车道中心线每段的中心点
              "num_nodes": 所有车道线段中点的总数
              "feats": 所有车道中心线每段的方向
              "turn": 所有车道的转向类型
              "control": 所有车道是否有红绿灯
              "intersect": 所有车道是否在路口
              "lane_idcs": 每个节点对应的车道编号（长度 = 总节点数， 节点是车道中心线线段）
            }
            'seq_id': 数据id(表明是哪条数据)
            ...
        }
        '''
        
        # visualization for debug purpose
        # self.visualize_data(data)
        return pd.DataFrame(
            [[data[key] for key in data.keys()]],
            columns=[key for key in data.keys()]
        )
    
    def save(self, dataframe: pd.DataFrame, file_name, dir_=None):
        if not isinstance(dataframe, pd.DataFrame):
            print('dataframe is not pd.DataFrame')
            return

        if not dir_:
            dir_ = os.path.join(os.path.split(self.raw_dir)[0], "intermediate", self.split + "_intermediate", "raw")
        else:
            dir_ = os.path.join(dir_, self.split + "_intermediate", "raw")

        if not os.path.exists(dir_):
            os.makedirs(dir_)

        fname = f"features_{file_name}.pkl"
        dataframe.to_pickle(os.path.join(dir_, fname))
        # print("[Preprocessor]: Saving data to {} with name: {}...".format(dir_, fname))

    @staticmethod
    def read_argo_data(df: pd.DataFrame):
        """TIMESTAMP, TRACK_ID, OBJECT_TYPE, X, Y, CITY_NAME"""

        # 获取城市
        city = df["CITY_NAME"].values[0]
        # 剔除重复的时间戳并进行排序（np.unique其实已经默认升序，np.sort只是稳妥起见）
        agt_ts = np.sort(np.unique(df['TIMESTAMP'].values))
        # 给每个时间戳一个索引
        mapping = dict()
        for i, ts in enumerate(agt_ts):
            mapping[ts] = i
        # 将轨迹合成为二维数组N行2列（-1表示自动计算行数，第二个参数 1 表示 沿列方向（横向）拼接）
        trajs = np.concatenate((
            df["X"].to_numpy().reshape(-1, 1),
            df["Y"].to_numpy().reshape(-1, 1)), 1)
        # 将时间戳转换为时间序列的索引（可能是乱序且重复的）
        steps = [mapping[x] for x in df['TIMESTAMP'].values]
        steps = np.asarray(steps, np.int64)
        # 找出所有的目标，并记录哪些行是某个目标的
        # objs是一个字典，键是[TRACK_ID， OBJECT_TYPE]，值是对应该目标的行
        objs = df.groupby(['TRACK_ID', 'OBJECT_TYPE']).groups
        # 得到所有的目标类型
        keys = list(objs.keys())
        obj_type = [x[1] for x in keys]
        # 获取Agent对应的数据行
        agt_idx = obj_type.index('AGENT')       
        idcs = objs[keys[agt_idx]]

        # 获取agent轨迹点，step表示每一行数据在时间序列里的索引（这里缺少去重和排序！！！）
        agt_traj = trajs[idcs]
        agt_step = steps[idcs]

        # 相当于删除AGENT数据，然后处理其他数据
        del keys[agt_idx]
        ctx_trajs, ctx_steps = [], []
        # 获取其他障碍物轨迹点
        for key in keys:
            idcs = objs[key]
            ctx_trajs.append(trajs[idcs])
            ctx_steps.append(steps[idcs])

        # 把agt_traj提出来就是为了将他放到最前面，可以通过列的交换实现
        # 记录下step可能是在后面的处理中用于去重和排序
        data = dict()
        data['city'] = city
        data['trajs'] = [agt_traj] + ctx_trajs
        data['steps'] = [agt_step] + ctx_steps
        return data

    def get_obj_feats(self, data):
        # 以Agent（也就是要预测的目标）的历史轨迹的最后一个点作为原点
        # 这个点相当于实际使用时Agent的当前位置，以此为坐标原点可以保证预测时各观测数据在尺度上的一致性与统一性
        orig = data['trajs'][0][self.obs_horizon-1].copy().astype(np.float32)

        # 计算旋转矩阵（此处分为旋转和不旋转，旋转就按车道方向，因为也没有Agent的角度）
        if self.normalized:
            # 获取原点对应车道中心线位置的切线向量及其置信度
            pre, conf = self.am.get_lane_direction(data['trajs'][0][self.obs_horizon-1], data['city'])
            if conf <= 0.1:
                pre = (orig - data['trajs'][0][self.obs_horizon-4]) / 2.0 # 只用了角度，没必要除以2
            theta = - np.arctan2(pre[1], pre[0]) + np.pi / 2  # 这个旋转不用加pi/2,但只要所有的数据统一加或不加就都可以
            rot = np.asarray([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]], np.float32)
        else:
            # if not normalized, do not rotate.
            theta = None
            rot = np.asarray([
                [1.0, 0.0],
                [0.0, 1.0]], np.float32)

        # 根据历史轨迹选出可能的车道中心线
        agt_traj_obs = data['trajs'][0][0: self.obs_horizon].copy().astype(np.float32)
        ctr_line_candts = self.am.get_candidate_centerlines_for_traj(agt_traj_obs, data['city'], viz=False)

        # 将未来轨迹和车道中心线变换到预测坐标系下（就是前面计算的原点和旋转矩阵）
        # agt_traj_obs应该也要转换吧？？？？
        agt_traj_fut = data['trajs'][0][self.obs_horizon:self.obs_horizon+self.pred_horizon].copy().astype(np.float32)
        agt_traj_fut = np.matmul(rot, (agt_traj_fut - orig.reshape(-1, 2)).T).T
        for i, _ in enumerate(ctr_line_candts):
            ctr_line_candts[i] = np.matmul(rot, (ctr_line_candts[i] - orig.reshape(-1, 2)).T).T

        # 采样候选目标点（基于转换后的中心线坐标），在每条中心线上从原点向前每隔0.5m采样一个点
        tar_candts = self.lane_candidate_sampling(ctr_line_candts, [0, 0], viz=False)

        # 如果是训练集和验证集，获取真值（最后一个历史轨迹点的车道、候选目标点及偏移值）
        if self.split == "test":
            tar_candts_gt, tar_offse_gt = np.zeros((tar_candts.shape[0], 1)), np.zeros((1, 2))
            splines, ref_idx = None, None
        else:
            # 获取真值对应的车道
            splines, ref_idx = self.get_ref_centerline(ctr_line_candts, agt_traj_fut)
            # 获取真值对应的候选点和offset
            tar_candts_gt, tar_offse_gt = self.get_candidate_gt(tar_candts, agt_traj_fut[-1])

        # 生成特征向量和标签
        feats, ctrs, has_obss, gt_preds, has_preds = [], [], [], [], []
        x_min, x_max, y_min, y_max = -self.obs_range, self.obs_range, -self.obs_range, self.obs_range
        '''
        data = {
            'trajs': [traj1, traj2, traj3, ...],
            'steps': [step1, step2, step3, ...],
            'city': 'MIA',
            ...
        }
        '''
        for traj, step in zip(data['trajs'], data['steps']):
            # traj表示一个目标的轨迹序列， step对应每个轨迹的时间步索引（时间戳对应的 step 索引）
            if self.obs_horizon-1 not in step:
                # 轨迹不够长的滤除
                continue

            # 对轨迹进行坐标转换，包括历史轨迹和未来轨迹（与前面的转换有点重复，浪费算力了！！！）
            traj_nd = np.matmul(rot, (traj - orig.reshape(-1, 2)).T).T

            # 获取预测轨迹真值
            # gt_pred 用于保存预测阶段的真实轨迹 (ground truth)
            gt_pred = np.zeros((self.pred_horizon, 2), np.float32)
            # has_pred 是一个布尔掩码，表示该时间步是否存在有效的 ground truth
            has_pred = np.zeros(self.pred_horizon, np.bool_)
            # 生成一个布尔掩码（mask），future_mask 为 True 的元素表示该点属于“预测阶段”
            future_mask = np.logical_and(
                step >= self.obs_horizon,
                step < self.obs_horizon + self.pred_horizon
            )
            # 取出这些未来帧对应的 step 值，并把时间归一化到(0, pred_horizon-1)
            post_step = step[future_mask] - self.obs_horizon
            # 取出这些未来帧对应的 (x, y) 坐标
            post_traj = traj_nd[future_mask]
            # 把这些真实轨迹放入 gt_pred 中对应的位置
            gt_pred[post_step] = post_traj
            # 标记这些时间步的预测目标存在
            has_pred[post_step] = True

            # 获取观测轨迹
            # 生成一个布尔掩码（mask），obs_mask 为 True 的元素表示该点属于观测阶段
            obs_mask = step < self.obs_horizon
            # 取出观测阶段的时间步
            step_obs = step[obs_mask]
            # 取出观测窗口的轨迹坐标
            traj_obs = traj_nd[obs_mask]
            # 有时候原始 CSV 中的时间戳不是升序的，所以这里进行排序
            idcs = step_obs.argsort()
            # 根据排序结果重排时间步和轨迹，保证按时间顺序排列
            step_obs = step_obs[idcs]
            traj_obs = traj_obs[idcs]

            # 用于对齐轨迹，但似乎有点奇怪（后续再细看！！！！）
            for i in range(len(step_obs)):
                if step_obs[i] == self.obs_horizon - len(step_obs) + i:
                    break
            step_obs = step_obs[i:]
            traj_obs = traj_obs[i:]

            if len(step_obs) <= 1:
                continue

            feat = np.zeros((self.obs_horizon, 3), np.float32)
            has_obs = np.zeros(self.obs_horizon, np.bool_)

            feat[step_obs, :2] = traj_obs
            feat[step_obs, 2] = 1.0
            has_obs[step_obs] = True

            if feat[-1, 0] < x_min or feat[-1, 0] > x_max or feat[-1, 1] < y_min or feat[-1, 1] > y_max:
                continue

            feats.append(feat)                  # displacement vectors
            has_obss.append(has_obs)
            gt_preds.append(gt_pred)
            has_preds.append(has_pred)

        # if len(feats) < 1:
        #     raise Exception()

        # feats是所有目标的观测轨迹
        feats = np.asarray(feats, np.float32)
        # has_obss标记所有目标的这些时间步的观测目标存在
        has_obss = np.asarray(has_obss, np.bool_)
        # gt_preds是所有目标的预测轨迹真值
        gt_preds = np.asarray(gt_preds, np.float32)
        # has_preds标记所有目标的这些时间步的预测目标存在
        has_preds = np.asarray(has_preds, np.bool_)

        # plot the splines
        # self.plot_reference_centerlines(ctr_line_candts, splines, feats[0], gt_preds[0], ref_idx)

        # # target candidate filtering
        # tar_candts = np.matmul(rot, (tar_candts - orig.reshape(-1, 2)).T).T
        # inlier = np.logical_and(np.fabs(tar_candts[:, 0]) <= self.obs_range, np.fabs(tar_candts[:, 1]) <= self.obs_range)
        # if not np.any(candts_gt[inlier]):
        #     raise Exception("The gt of target candidate exceeds the observation range!")

        data['orig'] = orig
        data['theta'] = theta
        data['rot'] = rot

        data['feats'] = feats
        data['has_obss'] = has_obss
        data['gt_preds'] = gt_preds
        data['has_preds'] = has_preds
        
        data['tar_candts'] = tar_candts
        data['gt_candts'] = tar_candts_gt
        data['gt_tar_offset'] = tar_offse_gt

        data['ref_ctr_lines'] = splines         # the reference candidate centerlines Spline for prediction
        data['ref_cetr_idx'] = ref_idx          # the idx of the closest reference centerlines
        return data

    def get_lane_graph(self, data):
        """Get a rectangle area defined by pred_range."""
        x_min, x_max, y_min, y_max = -self.obs_range, self.obs_range, -self.obs_range, self.obs_range
        radius = max(abs(x_min), abs(x_max)) + max(abs(y_min), abs(y_max))
        # 根据给定的二维坐标（或轨迹）和一个矩形范围（bbox），返回该范围内的所有车道 ID（lane_id）列表
        # box的长宽都是2 × query_search_range_m
        lane_ids = self.am.get_lane_ids_in_xy_bbox(data['orig'][0], data['orig'][1], data['city'], radius * 1.5)
        lane_ids = copy.deepcopy(lane_ids) # 单纯出于安全性和可读性考虑，是冗余的

        # 更新车道中心线和包络为预测坐标系下的坐标，并滤除观测范围外的车道
        lanes = dict()
        for lane_id in lane_ids:
            # 获取车道属性，city_lane_centerlines_dict返回的是城市/lane_id/道路属性这样层级的字典
            lane = self.am.city_lane_centerlines_dict[data['city']][lane_id]
            lane = copy.deepcopy(lane) # 单纯出于安全性和可读性考虑，是冗余的
            # 车道中心线转换到预测坐标系下
            centerline = np.matmul(data['rot'], (lane.centerline - data['orig'].reshape(-1, 2)).T).T
            # 冗余的保护措施，确保车道是在指定范围
            x, y = centerline[:, 0], centerline[:, 1]
            if x.max() < x_min or x.min() > x_max or y.max() < y_min or y.min() > y_max:
                continue
            else:
                """Getting polygons requires original centerline"""
                # 获取车道的包络点
                polygon = self.am.get_lane_segment_polygon(lane_id, data['city'])
                polygon = copy.deepcopy(polygon)
                # 将车道中心线和包络更新为坐标转换后的并添加到lanes里
                lane.centerline = centerline
                lane.polygon = np.matmul(data['rot'], (polygon[:, :2] - data['orig'].reshape(-1, 2)).T).T
                lanes[lane_id] = lane
        # 获取地图特征
        lane_ids = list(lanes.keys())
        ctrs, feats, turn, control, intersect = [], [], [], [], []
        for lane_id in lane_ids:
            lane = lanes[lane_id]
            # 车道中心线每段的中心点，方向
            ctrln = lane.centerline
            ctrs.append(np.asarray((ctrln[:-1] + ctrln[1:]) / 2.0, np.float32))
            feats.append(np.asarray(ctrln[1:] - ctrln[:-1], np.float32))
            # 车道的转向类型
            num_segs = len(ctrln) - 1 # 节点数是车道中心线线段的数量
            x = np.zeros((num_segs, 2), np.float32)
            if lane.turn_direction == 'LEFT':
                x[:, 0] = 1
            elif lane.turn_direction == 'RIGHT':
                x[:, 1] = 1
            else:
                pass
            turn.append(x)
            # 是否有红绿灯和是否在路口
            control.append(lane.has_traffic_control * np.ones(num_segs, np.float32))
            intersect.append(lane.is_intersection * np.ones(num_segs, np.float32))

        lane_idcs = []
        count = 0
        for i, ctr in enumerate(ctrs):
            lane_idcs.append(i * np.ones(len(ctr), np.int64))
            count += len(ctr)
        num_nodes = count # 所有车道线段中点的总数
        lane_idcs = np.concatenate(lane_idcs, 0)  # 每个节点对应的车道编号（长度 = 总节点数）

        graph = dict()
        graph['ctrs'] = np.concatenate(ctrs, 0)
        graph['num_nodes'] = num_nodes
        graph['feats'] = np.concatenate(feats, 0)
        graph['turn'] = np.concatenate(turn, 0)
        graph['control'] = np.concatenate(control, 0)
        graph['intersect'] = np.concatenate(intersect, 0)
        graph['lane_idcs'] = lane_idcs

        return graph

    @staticmethod
    def uniform_candidate_sampling(sampling_range, rate=30):
        """
        uniformly sampling of the target candidate
        :param sampling_range: int, the maximum range of the sampling
        :param rate: the sampling rate (num. of samples)
        return rate^2 candidate samples
        """
        x = np.linspace(-sampling_range, sampling_range, rate)
        return np.stack(np.meshgrid(x, x), -1).reshape(-1, 2)

    # implement a candidate sampling with equal distance;
    def lane_candidate_sampling(self, centerline_list, orig, distance=0.5, viz=False):
        """the input are list of lines, each line containing"""
        candidates = []
        # 分别对可能到达的车道进行候选点采样
        for lane_id, line in enumerate(centerline_list):
            # 车道中心线拟合三次样条曲线
            sp = Spline2D(x=line[:, 0], y=line[:, 1])
            # 计算坐标原点的纵向和横向位置（只需要纵向位置）
            s_o, d_o = sp.calc_frenet_position(orig[0], orig[1])
            # 纵向每0.5m采一个点
            s = np.arange(s_o, sp.s[-1], distance)
            # 计算每个采样点的x和y坐标
            ix, iy = sp.calc_global_position_online(s)
            # 拼成Nx2维
            candidates.append(np.stack([ix, iy], axis=1))
        # 将每条车道的候选点整合到一起成为N_total x 2,然后还要去重
        candidates = np.unique(np.concatenate(candidates), axis=0)

        if viz:
            fig = plt.figure(0, figsize=(8, 7))
            fig.clear()
            for centerline_coords in centerline_list:
                visualize_centerline(centerline_coords)
            plt.scatter(candidates[:, 0], candidates[:, 1], marker="*", c="g", alpha=1, s=6.0, zorder=15)
            plt.xlabel("Map X")
            plt.ylabel("Map Y")
            plt.axis("off")
            plt.title("No. of lane candidates = {}; No. of target candidates = {};".format(len(centerline_list), len(candidates)))
            plt.show()

        return candidates

    @staticmethod
    def get_candidate_gt(target_candidate, gt_target):
        """
        find the target candidate closest to the gt and output the one-hot ground truth
        :param target_candidate, (N, 2) candidates
        :param gt_target, (1, 2) the coordinate of final target
        """
        displacement = gt_target - target_candidate
        gt_index = np.argmin(np.power(displacement[:, 0], 2) + np.power(displacement[:, 1], 2))

        onehot = np.zeros((target_candidate.shape[0], 1))
        onehot[gt_index] = 1

        offset_xy = gt_target - target_candidate[gt_index]
        return onehot, offset_xy

    @staticmethod
    def get_ref_centerline(cline_list, pred_gt):
        if len(cline_list) == 1:
            return [Spline2D(x=cline_list[0][:, 0], y=cline_list[0][:, 1])], 0
        else:
            line_idx = 0
            ref_centerlines = [Spline2D(x=cline_list[i][:, 0], y=cline_list[i][:, 1]) for i in range(len(cline_list))]

            # search the closest point of the traj final position to each center line
            min_distances = []
            for line in ref_centerlines:
                xy = np.stack([line.x_fine, line.y_fine], axis=1)
                diff = xy - pred_gt[-1, :2]
                dis = np.hypot(diff[:, 0], diff[:, 1])
                min_distances.append(np.min(dis))
            line_idx = np.argmin(min_distances)
            return ref_centerlines, line_idx

    # 绘图
    @staticmethod
    def plot_target_candidates(candidate_centerlines, traj_obs, traj_fut, candidate_targets):
        fig = plt.figure(1, figsize=(8, 7))
        fig.clear()

        # plot centerlines
        for centerline_coords in candidate_centerlines:
            visualize_centerline(centerline_coords)

        # plot traj
        plt.plot(traj_obs[:, 0], traj_obs[:, 1], "x-", color="#d33e4c", alpha=1, linewidth=1, zorder=15)
        # plot end point
        plt.plot(traj_obs[-1, 0], traj_obs[-1, 1], "o", color="#d33e4c", alpha=1, markersize=6, zorder=15)
        # plot future traj
        plt.plot(traj_fut[:, 0], traj_fut[:, 1], "+-", color="b", alpha=1, linewidth=1, zorder=15)

        # plot target sample
        plt.scatter(candidate_targets[:, 0], candidate_targets[:, 1], marker="*", c="green", alpha=1, s=6, zorder=15)

        plt.xlabel("Map X")
        plt.ylabel("Map Y")
        plt.axis("off")
        plt.title("No. of lane candidates = {}; No. of target candidates = {};".format(len(candidate_centerlines),
                                                                                       len(candidate_targets)))
        # plt.show(block=False)
        # plt.pause(0.01)
        plt.show()

    def plot_reference_centerlines(self, cline_list, splines, obs, pred, ref_line_idx):
        fig = plt.figure(0, figsize=(8, 7))
        fig.clear()

        for centerline_coords in cline_list:
            visualize_centerline(centerline_coords)

        for i, spline in enumerate(splines):
            xy = np.stack([spline.x_fine, spline.y_fine], axis=1)
            if i == ref_line_idx:
                plt.plot(xy[:, 0], xy[:, 1], "--", color="r", alpha=0.7, linewidth=1, zorder=10)
            else:
                plt.plot(xy[:, 0], xy[:, 1], "--", color="b", alpha=0.5, linewidth=1, zorder=10)

        self.plot_traj(obs, pred)

        plt.xlabel("Map X")
        plt.ylabel("Map Y")
        plt.axis("off")
        plt.show()
        # plt.show(block=False)
        # plt.pause(0.5)

    def plot_traj(self, obs, pred, traj_id=None):
        assert len(obs) != 0, "ERROR: The input trajectory is empty!"
        traj_na = "t{}".format(traj_id) if traj_id else "traj"
        obj_type = "AGENT" if traj_id == 0 else "OTHERS"

        plt.plot(obs[:, 0], obs[:, 1], color=self.COLOR_DICT[obj_type], alpha=1, linewidth=1, zorder=15)
        plt.plot(pred[:, 0], pred[:, 1], "d-", color=self.COLOR_DICT[obj_type], alpha=1, linewidth=1, zorder=15)

        plt.text(obs[0, 0], obs[0, 1], "{}_s".format(traj_na))

        if len(pred) == 0:
            plt.text(obs[-1, 0], obs[-1, 1], "{}_e".format(traj_na))
        else:
            plt.text(pred[-1, 0], pred[-1, 1], "{}_e".format(traj_na))

    def visualize_data(self, data):
        """
        visualize the extracted data, and exam the data
        """
        fig = plt.figure(0, figsize=(8, 7))
        fig.clear()

        # visualize the centerlines
        lines_ctrs = data['graph']['ctrs']
        lines_feats = data['graph']['feats']
        lane_idcs = data['graph']['lane_idcs']
        for i in np.unique(lane_idcs):
            line_ctr = lines_ctrs[lane_idcs == i]
            line_feat = lines_feats[lane_idcs == i]
            line_str = (2.0 * line_ctr - line_feat) / 2.0
            line_end = (2.0 * line_ctr[-1, :] + line_feat[-1, :]) / 2.0
            line = np.vstack([line_str, line_end.reshape(-1, 2)])
            visualize_centerline(line)

        # visualize the trajectory
        trajs = data['feats'][:, :, :2]
        has_obss = data['has_obss']
        preds = data['gt_preds']
        has_preds = data['has_preds']
        for i, [traj, has_obs, pred, has_pred] in enumerate(zip(trajs, has_obss, preds, has_preds)):
            self.plot_traj(traj[has_obs], pred[has_pred], i)

        plt.xlabel("Map X")
        plt.ylabel("Map Y")
        plt.axis("off")
        plt.show()
        # plt.show(block=False)
        # plt.pause(0.5)

def ref_copy(data):
    if isinstance(data, list):
        return [ref_copy(x) for x in data]
    if isinstance(data, dict):
        d = dict()
        for key in data:
            d[key] = ref_copy(data[key])
        return d
    return data


if __name__ == "__main__":
    param_dict = {
    'raw_dir': 'dataset/raw_data_test',          # root directory stored the dataset
    'save_dir':  'dataset/interm_data_test',
    'batch_size': 1,
    'num_workers': 1,
    'algo': 'tnt',
    'obs_horizon': 20,          # the number of timestampe for observation
    'obs_range': 100,       # the observation range
    'pred_horizon': 30,     # the number of timestamp for prediction
    'normalized': True
    }
    
    small = False

    for split in ["train", "val", "test"]:
        # 构建DataSet
        argoverse_processor = ArgoversePreprocessor(param_dict=param_dict, split=split)
        # 构建DataLoader, ArgoversePreprocessor类继承了DataSet，所以作为DataLoader的输入
        loader = DataLoader(argoverse_processor,
                            batch_size=param_dict['batch_size'],
                            num_workers=param_dict['num_workers'],
                            shuffle=False,
                            pin_memory=False,
                            drop_last=False)

        for i, data in enumerate(tqdm(loader)):
            if small:
                if split == "train" and i >=0:
                    break
                elif split == "val" and i >= 50:
                    break
                elif split == "test" and i >= 50:
                    break
