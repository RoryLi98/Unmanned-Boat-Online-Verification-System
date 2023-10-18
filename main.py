import time
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from AStarPlanner import AStarPlanner
from RRTPlanner import RRTPlanner
from Vplanner import DWA
from random import randint, uniform

# from planner import AStarPlanner,RRTPlanner
# from localplanner import dwa

plt.rcParams["figure.figsize"] = [8.0, 8.0]
plt.rcParams["figure.autolayout"] = True
plt.rcParams["keymap.save"].remove("s")


from ConfigParseUtils import Config
conf = Config()
conf.LoadConf("setting.ini")
g_VisualDistance = conf.ReadData("setting","VisualDistance",type='Int')
g_SelfDirection = conf.ReadData("setting","SelfDirection",type='Bool')
g_RelativeCoordinates = conf.ReadData("setting","RelativeCoordinates",type='Bool')
g_TargetRadius = conf.ReadData("setting","TargetRadius",type='Float')
g_CheckPointRadius = conf.ReadData("setting","CheckPointRadius",type='Float')

import time

current_time = time.strftime("%Y%m%d%H%M%S")
# print(current_time)

def transformation_matrix(x, y, theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), x],
        [np.sin(theta), np.cos(theta), y],
        [0, 0, 1],
    ])

def split_into_parts(number, n):
    if n <= 0:
        return []

    quotient = number // n
    remainder = number % n

    parts = []
    start_index = 0

    for i in range(n):
        end_index = start_index + quotient + (1 if i < remainder else 0)
        parts.append(end_index - 1)
        start_index = end_index

    return parts


class DWAConfig:
    def __init__(self, obs_radius):
        self.obs_radius = obs_radius
        self.robot_radius = 0.5 # 艇半径
        self.safety_ratio = 1.5 # RRT和DWA的避障距离 = 障碍物半径 + 艇半径*安全距离设置倍数
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

    def __init__(self, planner=None, vplanner=None):
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.vx = 0.0
        self.vw = 0.0
        self.x_traj = []
        self.y_traj = []

        self.dwaconfig = DWAConfig(self.planning_obs_radius)
        self.dt = self.dwaconfig.dt

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

        self.planning_path = np.empty(shape=(0, 2))
        self.planning_target = None

        self.planner = planner
        self.vplanner = vplanner
        self.vplanner_midpos_index = None

        #####################################
        self.temp_obs = [0, 0]
        self.temp_targetPos = np.empty(shape=(0, 2))
        self.lastStage = -1
        self.curStage = -1
        self.finalTarget = None
        self.checkpoints = np.empty(shape=(0, 2))

    def run(self):
        while True:
            if (self.temp_targetPos.shape[0] != 0 and self.curStage < self.temp_targetPos.shape[0]-1):
                if(self.curStage == -1 or self.curStage == self.lastStage):
                    self.lastStage = self.curStage
                    self.curStage += 1

                    if(self.curStage == self.temp_targetPos.shape[0]-1):
                        self.planning_target = self.finalTarget
                        self.vplanner.passFlag = False
                    else:
                        self.planning_target = self.temp_targetPos[self.curStage]
                    px, py = planner.planning(self.planning_obs[:, 0], self.planning_obs[:, 1], Playground.planning_obs_radius + self.dwaconfig.robot_radius * self.dwaconfig.safety_ratio, self.x, self.y, self.planning_target[0], self.planning_target[1], -10, -10, 10, 10)
                    self.planning_path = np.vstack([px, py]).T
                    self.vplanner_midpos_index = None

            if self.NEED_EXIT:
                plt.close("all")
                break
            
            self.vplanner_midpos_index = self.check_path()
            all_traj = []
            all_u = []
            best_traj = None
            if self.vplanner_midpos_index >= 0:
                self.UpdateDynamicObstacle()
                all_planning_obs = np.append(self.planning_obs, self.dynamic_planning_obs, axis=0)
                # 选择全局路径规划中的点作为局部规划的终点
                midpos = self.planning_path[self.vplanner_midpos_index]
                # print(midpos)
                [self.vx,self.vw], best_traj, all_traj, all_u = self.vplanner.plan([self.x, self.y, self.theta, self.vx, self.vw], self.dwaconfig, midpos, all_planning_obs)

                # print(f"TarDis:{np.sqrt((self.x-self.planning_target[0]) ** 2 + (self.y-self.planning_target[1]) ** 2)}")
                if (np.sqrt((self.x-self.planning_target[0]) ** 2 + (self.y-self.planning_target[1]) ** 2) < g_TargetRadius):
                    plt.savefig(F'{current_time}-S.png')  # 保存为PNG格式
                    break
                else:
                    plt.savefig(F'{current_time}-F.png')  # 保存为PNG格式
            else:
                self.vx, self.vw = 0.0, 0.0

            dx, dy, dw = self.vx * self.dt, 0, self.vw * self.dt
            T = transformation_matrix(self.x, self.y, self.theta)
            p = np.matmul(T, np.array([dx, dy, 1]))
            self.x = p[0]
            self.y = p[1]
            self.theta += dw
            self.x_traj.append(self.x)
            self.y_traj.append(self.y)

            plt.cla()
            self.__draw(all_traj, all_u, best_traj=best_traj)
            
            if (self.temp_targetPos.shape[0] != 0):
                # self.ax.plot(self.planning_target[0], self.planning_target[1], "o", markersize=30)
                # self.ax.plot(self.finalTarget[0], self.finalTarget[1], "g+", markersize=30)
                # print(F"Chasing {self.planning_target}...") # 当前正在追踪哪个点
                if(np.sqrt((self.x-self.planning_target[0]) ** 2 + (self.y-self.planning_target[1]) ** 2) < g_CheckPointRadius):
                    self.checkpoints = np.append(self.checkpoints, [[self.planning_target[0],self.planning_target[1]]], axis=0)
                    # print(self.checkpoints)
                    self.lastStage +=1

    def check_path(self):
        if self.planning_path is None or self.planning_path.shape[0] == 0:
            return -1
        if self.vplanner_midpos_index is not None and self.vplanner_midpos_index >= 0:
            midindex = self.vplanner_midpos_index
            while True:
                midpos = self.planning_path[midindex]
                dist = np.hypot(self.x - midpos[0], self.y - midpos[1])
                if dist > self.dwaconfig.tracking_dist:
                    break
                if midindex + 1 == self.planning_path.shape[0]:
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

    def __draw(self, all_traj, all_value, best_traj):
        # assert(self.planning_path is None or self.planning_path.shape[1] == 2,
        #        "the shape of planning path should be '[x,2]', please check your algorithm.")
        # assert(self.planning_obs is None or self.planning_obs.shape[1] == 2,
        #        "the shape of self.planning_obs(obstacles) should be '[x,2]', please check your algorithm.")

        p1_i = np.array([0.5, 0, 1]).T
        p2_i = np.array([-0.5, 0.25, 1]).T
        p3_i = np.array([-0.5, -0.25, 1]).T

        T = transformation_matrix(self.x, self.y, self.theta)
        p1 = np.matmul(T, p1_i)
        p2 = np.matmul(T, p2_i)
        p3 = np.matmul(T, p3_i)

        plt.plot([p1[0], p2[0], p3[0], p1[0]], [p1[1], p2[1], p3[1], p1[1]], "k-")

        if self.planning_target is not None:
            # self.ax.plot(self.planning_target[0], self.planning_target[1], "rx", markersize=20)
            self.ax.plot(self.planning_target[0], self.planning_target[1], "ro", markersize=5)

        if self.finalTarget is not None:
            self.ax.plot(self.finalTarget[0], self.finalTarget[1], "gx", markersize=20)

        for checkpointPos in self.checkpoints:
            self.ax.plot(checkpointPos[0], checkpointPos[1], "bx", markersize=10)

        if len(all_traj) > 0:
            all_value = np.array(all_value, dtype=float)
            maxValue = -1
            if all_value.max() == float("Inf"):
                maxValue = 10000
            else:
                maxValue = all_value.max()
            all_value = (all_value - all_value.min()) / (maxValue - all_value.min())  # 归一化[0-1]
            for i, traj in enumerate(all_traj):
                color = plt.cm.jet(all_value[i])
                self.ax.plot(traj[:, 0],  traj[:, 1],  ".", color=color, markersize=1)
                # self.ax.plot(traj[-1,0],traj[-1,1],"+",color=color,markersize=3)

        if best_traj is not None:
            self.ax.plot(best_traj[:, 0], best_traj[:, 1], color="green", linewidth=3)

        if self.planning_path is not None:
            # self.ax.plot(self.planning_path[:, 0], self.planning_path[:, 1], "b--")
            if (self.vplanner_midpos_index is not None and self.vplanner_midpos_index >= 0):
                midpos = self.planning_path[self.vplanner_midpos_index]
                # self.ax.plot(midpos[0], midpos[1], "g+", markersize=20)  # 隐藏规划轨迹

        if len(self.x_traj) > 0:  # 真实轨迹
            if self.startDraw:
                plt.plot(self.x_traj, self.y_traj, "g-", markersize=10)
                
        for obs in self.planning_obs:
            self.ax.add_artist(
                plt.Circle((obs[0], obs[1]), self.planning_obs_radius, fill=True, color="blue"))

        for obs in self.dynamic_planning_obs:
            self.ax.add_artist(
                plt.Circle((obs[0], obs[1]), self.planning_obs_radius, fill=True, color="red"))

        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)

        plt.pause(self.dt)  # 暂停最小单位时间


    def on_mousepress(self, event):
        if not event.dblclick:
            if event.button == 1:  # 左键设置起点
                self.x, self.y = event.xdata, event.ydata
            if event.button == 3:  # 右键设置终点
                self.planning_target = np.array([event.xdata, event.ydata])
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
            self.planning_path = None
            self.x_traj, self.y_traj = [], []
            self.vplanner_midpos_index = None
            if self.planning_target is not None and self.planner is not None:
                print("do planning...")
                self.finalTarget = self.planning_target.copy()
                print(self.planning_obs.shape)
                px, py = planner.planning(self.planning_obs[:, 0], self.planning_obs[:, 1], Playground.planning_obs_radius+ self.dwaconfig.robot_radius * self.dwaconfig.safety_ratio, self.x, self.y, self.planning_target[0], self.planning_target[1], -10, -10, 10, 10)
                self.planning_path = np.vstack([px, py]).T
                print("pathLength : ", self.planning_path.shape[0])

                ids = split_into_parts(self.planning_path.shape[0], 5)
                for id in ids:
                    self.temp_targetPos = np.append(self.temp_targetPos, [[self.planning_path[id][0], self.planning_path[id][1]]], axis=0)


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

    pg = Playground(planner, vplanner)
    pg.run()
