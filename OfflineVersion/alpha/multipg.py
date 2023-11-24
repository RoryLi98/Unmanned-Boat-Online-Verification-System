import time
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from AStarPlanner import AStarPlanner
from RRTPlanner import RRTPlanner
from Vplanner import DWA
from random import randint, uniform
import copy

# from planner import AStarPlanner,RRTPlanner
# from localplanner import dwa

plt.rcParams["figure.figsize"] = [8.0, 8.0]
plt.rcParams["figure.autolayout"] = True
plt.rcParams["keymap.save"].remove("s")


def transformation_matrix(x, y, theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), x],
        [np.sin(theta), np.cos(theta), y],
        [0, 0, 1],
    ])


class DWAConfig:

    def __init__(self, obs_radius):
        self.obs_radius = obs_radius
        self.robot_radius = 0.5 # 艇半径
        self.safety_ratio = 0.8 # RRT和DWA的避障距离 = 障碍物半径 + 艇半径*安全距离设置倍数
        self.dt = 0.1  # [s] Time tick for motion prediction

        self.max_speed = 1.5  # [m/s] 最大线速度
        self.min_speed = -0.5  # [m/s] 最小线速度
        self.max_accel = 0.5  # [m/ss] 加速度
        self.v_reso = self.max_accel * self.dt / 10.0  # [m/s] 速度增加的步长

        self.min_yawrate = -100.0 * math.pi / 180.0  # [rad/s] 最小角速度
        self.max_yawrate = 100.0 * math.pi / 180.0  # [rad/s] 最大角速度
        self.max_dyawrate = 300.0 * math.pi / 180.0  # [rad/ss] 角加速度
        self.yawrate_reso = self.max_dyawrate * self.dt / 10.0  # [rad/s] 角速度增加的步长

        # 模拟轨迹的持续时间
        self.predict_time = 2.5  # [s]  

        # 三个比例系数
        self.to_goal_cost_gain = 3  # 距离目标点的评价函数的权重系数
        self.speed_cost_gain = 2 # 速度评价函数的权重系数
        self.obstacle_cost_gain = 1  # 距离障碍物距离的评价函数的权重系数

        self.tracking_dist = self.predict_time * self.max_speed  # 自动局部避障终点
        self.arrive_dist = 0.1


class Playground:
    planning_obs_radius = 0.5

    def __init__(self, planner=None, vplanner=None, v_num=3):
        self.v_num = v_num
        self.x = []
        self.y = []
        self.theta = []
        self.vx = []
        self.vw = []

        self.dwaconfigs = []
        self.dts = []
        self.init_traj = []
        for i in range(0, v_num):
            self.dwaconfigs.append(DWAConfig(self.planning_obs_radius))
            self.dts.append(self.dwaconfigs[i].dt)
            self.x.append(0.0)
            self.y.append(0.0)
            self.theta.append(0.0)
            self.vx.append(0.0)
            self.vw.append(0.0)
            self.init_traj.append([])
        self.x_traj = copy.deepcopy(self.init_traj)
        self.y_traj = copy.deepcopy(self.init_traj)

        self.fig, self.ax = plt.subplots()

        self.fig.canvas.mpl_connect("button_press_event", self.on_mousepress)
        self.fig.canvas.mpl_connect("key_press_event", self.on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_mousemove)
        self.NEED_EXIT = False
        self.startDraw = False

        ############################################

        self.planning_obs = np.empty(shape=(0, 2))
        self.dynamic_planning_obs = np.empty(shape=(0, 2))
        self.dynamic_planning_obs_range = 0.1

        self.planning_paths = []
        self.vplanner_midpos_indexs = []

        for i in range(0, v_num):
            self.planning_paths.append(np.empty(shape=(0, 2)))
            self.vplanner_midpos_indexs.append(None)
        self.planning_target = None

        self.planner = planner
        self.vplanner = vplanner


        #####################################
        self.temp_obs = [0, 0]



    def run(self):
        while True:
            if self.NEED_EXIT:
                plt.close("all")
                break

            all_trajs = []
            all_us = []
            best_trajs = []
            for v_id in range(0, self.v_num):
                self.vplanner_midpos_indexs[v_id] = self.check_path(v_id)
                all_traj = []
                all_u = []
                best_traj = None
                if self.vplanner_midpos_indexs[v_id] >= 0:
                    if v_id == 0:
                        self.UpdateDynamicObstacle()
                    other_vs = np.empty(shape=(0, 2))
                    for other_v_id in range(0, self.v_num):
                        if other_v_id != v_id:
                            other_vs = np.append(other_vs, [[self.x[other_v_id], self.y[other_v_id]]], axis=0)
                    all_planning_obs = np.concatenate([other_vs, self.planning_obs, self.dynamic_planning_obs], axis=0)
                    # 选择全局路径规划中的点作为局部规划的终点
                    midpos = self.planning_paths[v_id][self.vplanner_midpos_indexs[v_id]]
                    [self.vx[v_id], self.vw[v_id]], best_traj, all_traj, all_u = self.vplanner.plan(
                        [self.x[v_id], self.y[v_id], self.theta[v_id], self.vx[v_id], self.vw[v_id]], self.dwaconfigs[v_id], midpos, all_planning_obs)
                    if best_traj is None:
                        print("Broken boat: ", v_id)
                        time.sleep(100)
                        self.NEED_EXIT = True
                else:
                    self.vx[v_id], self.vw[v_id] = 0.0, 0.0

                dx, dy, dw = self.vx[v_id] * self.dts[v_id], 0, self.vw[v_id] * self.dts[v_id]
                T = transformation_matrix(self.x[v_id], self.y[v_id], self.theta[v_id])
                p = np.matmul(T, np.array([dx, dy, 1]))
                self.x[v_id] = p[0]
                self.y[v_id] = p[1]
                self.theta[v_id] += dw
                self.x_traj[v_id].append(self.x[v_id])
                self.y_traj[v_id].append(self.y[v_id])

                all_trajs.append(all_traj)
                all_us.append(all_u)
                best_trajs.append(best_traj)

            plt.cla()
            self.__draw(all_trajs, all_us, best_trajs=best_trajs)
            # self.vplanner_midpos_index = self.check_path()
            # all_traj = []
            # all_u = []
            # best_traj = None
            # if self.vplanner_midpos_index >= 0:
            #     self.UpdateDynamicObstacle()
            #     all_planning_obs = np.append(self.planning_obs, self.dynamic_planning_obs, axis=0)
            #     # 选择全局路径规划中的点作为局部规划的终点
            #     midpos = self.planning_path[self.vplanner_midpos_index]
            #     [self.vx, self.vw], best_traj, all_traj, all_u = self.vplanner.plan(
            #         [self.x, self.y, self.theta, self.vx, self.vw], self.dwaconfig, midpos, all_planning_obs)
            # else:
            #     self.vx, self.vw = 0.0, 0.0
            #
            # dx, dy, dw = self.vx * self.dt, 0, self.vw * self.dt
            # T = transformation_matrix(self.x, self.y, self.theta)
            # p = np.matmul(T, np.array([dx, dy, 1]))
            # self.x = p[0]
            # self.y = p[1]
            # self.theta += dw
            # self.x_traj.append(self.x)
            # self.y_traj.append(self.y)
            #
            # plt.cla()
            # self.__draw(all_traj, all_u, best_traj=best_traj)

    def check_path(self, v_id):
        if self.planning_paths[v_id] is None or self.planning_paths[v_id].shape[0] == 0:
            return -1
        if self.vplanner_midpos_indexs[v_id] is not None and self.vplanner_midpos_indexs[v_id] >= 0:
            midindex = self.vplanner_midpos_indexs[v_id]
            while True:
                midpos = self.planning_paths[v_id][midindex]
                dist = np.hypot(self.x[v_id] - midpos[0], self.y[v_id] - midpos[1])
                if dist > self.dwaconfigs[v_id].tracking_dist:
                    break
                if midindex + 1 == self.planning_paths[v_id].shape[0]:
                    return midindex
                midindex += 1
            return midindex
        else:
            return 0

    def add_obs(self, x, y):
        self.planning_obs = np.append(self.planning_obs, [[x, y]], axis=0)

    def add_dynamicObs(self, x, y):
        self.dynamic_planning_obs = np.append(self.dynamic_planning_obs, [[x, y]], axis=0)

    def add_obss(self, xs, ys):
        self.planning_obs = np.append(self.planning_obs, np.vstack([xs, ys]).T, axis=0)

    def __draw(self, all_trajs, all_values, best_trajs):
        # assert(self.planning_path is None or self.planning_path.shape[1] == 2,
        #        "the shape of planning path should be '[x,2]', please check your algorithm.")
        # assert(self.planning_obs is None or self.planning_obs.shape[1] == 2,
        #        "the shape of self.planning_obs(obstacles) should be '[x,2]', please check your algorithm.")

        if self.planning_target is not None:
            self.ax.plot(self.planning_target[0], self.planning_target[1], "rx", markersize=30)

        for v_id in range(0, self.v_num):
            p1_i = np.array([0.5, 0, 1]).T
            p2_i = np.array([-0.5, 0.25, 1]).T
            p3_i = np.array([-0.5, -0.25, 1]).T

            T = transformation_matrix(self.x[v_id], self.y[v_id], self.theta[v_id])
            p1 = np.matmul(T, p1_i)
            p2 = np.matmul(T, p2_i)
            p3 = np.matmul(T, p3_i)

            plt.plot([p1[0], p2[0], p3[0], p1[0]], [p1[1], p2[1], p3[1], p1[1]], "k-")

            if len(all_trajs[v_id]) > 0:
                all_values[v_id] = np.array(all_values[v_id], dtype=float)
                maxValue = -1
                if all_values[v_id].max() == float("Inf"):
                    maxValue = 10000
                else:
                    maxValue = all_values[v_id].max()
                all_values[v_id] = (all_values[v_id] - all_values[v_id].min()) / (maxValue - all_values[v_id].min())  # 归一化[0-1]
                for i, traj in enumerate(all_trajs[v_id]):
                    color = plt.cm.jet(all_values[v_id][i])
                    self.ax.plot(traj[:, 0], traj[:, 1], ".", color=color, markersize=1)
                    # self.ax.plot(traj[-1,0],traj[-1,1],"+",color=color,markersize=3)

            if best_trajs[v_id] is not None:
                self.ax.plot(best_trajs[v_id][:, 0], best_trajs[v_id][:, 1], color="green", linewidth=3)

            planning_path = self.planning_paths[v_id]
            if planning_path is not None:
                self.ax.plot(planning_path[:, 0], planning_path[:, 1], "b--")
                if (self.vplanner_midpos_indexs[v_id] is not None and self.vplanner_midpos_indexs[v_id] >= 0):
                    midpos = planning_path[self.vplanner_midpos_indexs[v_id]]
                    self.ax.plot(midpos[0], midpos[1], "g+", markersize=20)
            if len(self.x_traj[v_id]) > 0:  # 真实轨迹
                if self.startDraw:
                    plt.plot(self.x_traj[v_id], self.y_traj[v_id], "g-", markersize=10)

        for obs in self.planning_obs:
            self.ax.add_artist(
                plt.Circle((obs[0], obs[1]), self.planning_obs_radius, fill=True, color="blue"))

        for obs in self.dynamic_planning_obs:
            self.ax.add_artist(
                plt.Circle((obs[0], obs[1]), self.planning_obs_radius, fill=True, color="red"))

        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)

        plt.pause(self.dts[0])  # 暂停最小单位时间

    def on_mousepress(self, event):
        if not event.dblclick:
            # if event.button == 1:  # 左键设置起点
            #     if len(self.x) < self.v_num:
            #         self.x.append(event.xdata)
            #         self.y.append(event.ydata)
            #     else:
            #         pass
                # self.x, self.y = event.xdata, event.ydata
            if event.button == 3:  # 右键设置终点
                self.planning_target = np.array([event.xdata, event.ydata])
                
            if event.button == 1:  # 单击左键添加单个静态障碍
                for i in range(0, self.v_num):
                    self.theta[i] = math.atan2(self.planning_target[1]-self.x[i], self.planning_target[0]-self.y[i])
                    
            if event.button == 2:  # 单击中键添加单个静态障碍
                self.add_obs(event.xdata, event.ydata)
                self.temp_obs = [event.xdata, event.ydata]

    def on_mousemove(self, event):
        if hasattr(event, "button") and event.button == 2:  # 持续的中键添加连续静态障碍
            dx = event.xdata - self.temp_obs[0]
            dy = event.ydata - self.temp_obs[1]
            if np.hypot(dx, dy) > self.planning_obs_radius * 0.8:  # 障碍点之间的距离不能小于0.8的半径
                self.temp_obs = [event.xdata, event.ydata]
                self.add_obs(*self.temp_obs)

    def on_press(self, event):
        if event.key == "m":  # 添加动态障碍
            self.add_dynamicObs(event.xdata, event.ydata)
            self.temp_obs = [event.xdata, event.ydata]
        if event.key == "escape":  # ESC退出
            self.set_exit()
        if event.key == " ":  # 空格
            self.startDraw = True

            self.x_traj, self.y_traj = copy.deepcopy(self.init_traj), copy.deepcopy(self.init_traj)
            for v_id in range(0, self.v_num):
                self.planning_paths += [None]
                self.vplanner_midpos_indexs += [None]
            if self.planning_target is not None and self.planner is not None:
                print("do planning...")
                for v_id in range(0, self.v_num):
                    all_obsx = copy.deepcopy(self.planning_obs[:, 0])
                    all_obsy = copy.deepcopy(self.planning_obs[:, 1])
                    if v_id > 0:
                        pre_path = []
                        for pre in range(0, v_id):
                            pre_path.append(self.planning_paths[pre][0:-100])
                        # print("pre_path", pre_path)
                        for pre_poss in pre_path:
                            for pre_pos in pre_poss:
                                print("pre_pos[0]:", pre_pos[0])
                                print("all_obsx:", all_obsx)
                                all_obsx = np.insert(all_obsx, -1, pre_pos[0], axis=0)
                                all_obsy = np.insert(all_obsy, -1, pre_pos[1], axis=0)
                    px, py = planner.planning(all_obsx, all_obsy,
                                              Playground.planning_obs_radius + self.dwaconfigs[v_id].robot_radius * self.dwaconfigs[v_id].safety_ratio,
                                              self.x[v_id], self.y[v_id], self.planning_target[0], self.planning_target[1], -10, -10,
                                              10, 10)
                    planning_path = np.vstack([px, py]).T
                    self.planning_paths[v_id] = planning_path
                    print(v_id, "pathLength : ", planning_path.shape[0])
# <<<<<<< Updated upstream
#
# =======
#                     # print("planning_path: ", planning_path)
#                     # print("planning_obs: ", self.planning_obs)
#                     # print("planning_obs[:, 0]", self.planning_obs[:, 0])
# >>>>>>> Stashed changes
        if event.key in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:  # 添加无人艇
            if int(event.key) in range(0, self.v_num):
                self.x[int(event.key)], self.y[int(event.key)] = event.xdata, event.ydata
                if self.planning_target is not None:
                    self.theta[int(event.key)] = math.atan2(self.planning_target[1]-event.ydata, self.planning_target[0]-event.xdata)

        # if event.key in range(0, self.v_num):  # 添加动态障碍
        #     self.x[event.key], self.y[event.key] = event.xdata, event.ydata

    def set_exit(self):
        self.NEED_EXIT = True

    def UpdateDynamicObstacle(self):
        for i in self.dynamic_planning_obs:
            i[0] += uniform(-self.dynamic_planning_obs_range, self.dynamic_planning_obs_range)
            i[1] += uniform(-self.dynamic_planning_obs_range, self.dynamic_planning_obs_range)


if __name__ == "__main__":
    planner = None
    planner = AStarPlanner(0.2)
    # planner = RRTPlanner(0.2)
    vplanner = DWA()

    pg = Playground(planner, vplanner, 3)
    pg.run()
