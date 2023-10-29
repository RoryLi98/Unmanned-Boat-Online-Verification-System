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

import openai
import numpy as np
import json
import re

def find_two_floats(input_string):
    # 去掉第一个 "(" 之前的内容
    start_index = input_string.find('(')
    if start_index != -1:
        input_string = input_string[start_index + 1:]

    # 去掉第一个 ")" 后面的内容
    end_index = input_string.find(')')
    if end_index != -1:
        input_string = input_string[:end_index]
    
    # 去除空格
    input_string = input_string.replace(" ", "")

    # 找到逗号的位置
    comma_index = input_string.index(",")

    # 提取逗号前后的两个子字符串
    num1_str = input_string[:comma_index]
    num2_str = input_string[comma_index + 1:]

    # 将子字符串转换为浮点数
    num1 = float(num1_str)
    num2 = float(num2_str)

    return num1, num2


from ConfigParseUtils import Config
conf = Config()
conf.LoadConf("setting.ini")
g_VisualDistance = conf.ReadData("setting","VisualDistance",type='Int')             # 可视距离
g_SelfDirection = conf.ReadData("setting","SelfDirection",type='Bool')              # 是否以船头方向为正方向
g_RelativeCoordinates = conf.ReadData("setting","RelativeCoordinates",type='Bool')  # 是否用相对坐标
g_TargetRadius = conf.ReadData("setting","TargetRadius",type='Float')               #
g_CheckPointRadius = conf.ReadData("setting","CheckPointRadius",type='Float')
g_RoundPlace = conf.ReadData("setting","RoundPlace",type='Int')
g_ObsRadius = conf.ReadData("setting","ObsRadius",type='Float')
g_SafeRadius = 2*g_ObsRadius

### 加载配置文件 获取API
with open("config.json", "r") as f:
    config = json.load(f)

openai.api_base = config["OPENAI_API_BASE"]
openai.api_key = config["OPENAI_API_KEY"]

chat_history = [
    {
        "role": "system",
        "content": "---"
    },
    {
        "role": "user",
        "content": "---"
    },
    {
        "role": "assistant",
        "content": "---"
    }
]

def ask(question):
    chat_history.append(
        {
            "role": "user",
            "content": question,
        }
    )
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=chat_history,
        temperature=0
    )
    chat_history.append(
        {
            "role": "assistant",
            "content": completion.choices[0].message.content,
        }
    )
    return chat_history[-1]["content"]

first_instruction = \
'''You are the command system of an unmanned boat, and your task is to direct the unmanned boat to avoid obstacles and complete the task objectives according to the environmental information returned by the unmanned boat perception system. If you understand your assignment, answer yes, otherwise answer no.'''
# findNextCheckpoint = \
# '''Next I will tell you the environmental information: \nCurrent coordinates of the ship: CURPOS Rescue target: TARPOS \nLocation of the reef: ENVIRONMENTSTATUS \nReef shape: a circle with a radius of OBSRADIUS.
# Your task is to guide the unmanned boat to reach the 'target position' quickly and safely while maintaining a distance from each reef during the journey. Considering the limited effective sensing range of the unmanned boat, please provide the coordinates for the next step within a 10 radius. Each step should bring the boat closer to the target, while staying at least SAFERADIUS away from each reef.
# When the distance between the "current position" and the "target position" is less than SAFERADIUS, the "next position" is the "target position."
# Please provide the next coordinates in the following format:
# The next position is:

# After providing the next position coordinates, please verify by calculation. if the distance from each reef is greater than SAFERADIUS. If it is less than SAFERADIUS, please replan and provide new coordinate positions. Please calculate the distance between the "current position" and "target position". If it is less than SAFERADIUS, "the next position" is the "target position".'''

findNextCheckpoint = \
'''Next I will tell you the environmental information(In the given coordinate, a unit of 1 represents 100 meters in reality): \nCurrent coordinates of the ship: CURPOS Rescue target: TARPOS \nLocation of the reef: ENVIRONMENTSTATUS \nReef shape: a circle with a radius of OBSRADIUS.
Your task is to guide the unmanned boat to reach the 'target position' quickly and safely while maintaining a distance from each reef during the journey. Each step should bring the boat closer to the target, while staying at least 1 away from each reef. Note that The farthest distance we can reach in a single move is 3.
Please provide the next coordinates in the following format:
The next position is:

After providing the next position coordinates, Please verify by calculation. If the distance from each reef is greater than 0.5. If it is less than 0.5, please give a new position and provide new coordinate positions. Please calculate the distance between "current position" and "target position". If it is less than 2 , "the next position" is "target position"'''

findNextCheckpoint = findNextCheckpoint.replace("OBSRADIUS",str(g_ObsRadius))
findNextCheckpoint = findNextCheckpoint.replace("SAFERADIUS",str(g_SafeRadius))


# print(findNextCheckpoint)
### 获取回复内容
response = ask(first_instruction)
print(first_instruction)
print(response)

import time

current_time = time.strftime("%Y%m%d%H%M%S")
# print(current_time)
def rotate_coordinate(x, y, theta):
    # 计算旋转后的坐标
    relTheta = 90*math.pi/180 - theta
    x_rotated = x * np.cos(relTheta) - y * np.sin(relTheta)
    y_rotated = x * np.sin(relTheta) + y * np.cos(relTheta)
    return x_rotated, y_rotated

def inverse_rotate_coordinate(x, y, theta):
    relTheta = theta - 90*math.pi/180
    x_rotated = x * np.cos(relTheta) - y * np.sin(relTheta)
    y_rotated = x * np.sin(relTheta) + y * np.cos(relTheta)
    return x_rotated, y_rotated

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
        self.obstacle_cost_gain = 0  # 距离障碍物距离的评价函数的权重系数

        self.tracking_dist = self.predict_time * self.max_speed  # 自动局部避障终点
        self.arrive_dist = 0.1


class Playground:
    planning_obs_radius = 0.5

    def __init__(self, planner=None, vplanner=None):
        self.x = 0.0
        self.y = 0.0
        self.theta = 0  # 0.0 90*math.pi/180
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
        self.lastStage = -1
        self.curStage = -1
        self.finalTarget = None
        self.checkpoints = np.empty(shape=(0, 2))
        self.arrivedCheckpoint = False
        self.chasingFlag = False

    def run(self):
        while True:
            if self.finalTarget is not None and self.chasingFlag and self.vplanner.passFlag:
                if ((self.arrivedCheckpoint) or (self.planning_target is None)):
                    # print(np.sqrt((self.x-self.finalTarget[0]) ** 2 + (self.y-self.finalTarget[1]) ** 2))
                    # if (np.sqrt((self.x-self.finalTarget[0]) ** 2 + (self.y-self.finalTarget[1]) ** 2) < g_TargetRadius):
                    #     plt.savefig(F'{current_time}-S.png')  # 保存为PNG格式
                    #     break
                    self.arrivedCheckpoint = False

                    seenObs = []
                    global g_VisualDistance
                    if(g_VisualDistance == -1):
                        g_VisualDistance = float("Inf")
                    
                    for ob in self.planning_obs:
                        if(np.sqrt((ob[0]-self.x) ** 2 + (ob[1]-self.y) ** 2) < g_VisualDistance):  #是否在可视距离内
                            tmpOb = ob.copy()
                            if(g_RelativeCoordinates):
                                tmpOb[0] = round(tmpOb[0]-self.x, g_RoundPlace)
                                tmpOb[1] = round(tmpOb[1]-self.y, g_RoundPlace)
                            if(g_SelfDirection):
                                if(not g_RelativeCoordinates):
                                    tmpOb[0] = round(tmpOb[0]-self.x, g_RoundPlace)
                                    tmpOb[1] = round(tmpOb[1]-self.y, g_RoundPlace)
                                tmpOb[0],tmpOb[1] = rotate_coordinate(tmpOb[0], tmpOb[1], self.theta)
                                # tmpOb[0],tmpOb[1] = inverse_rotate_coordinate(tmpOb[0], tmpOb[1], self.theta)
                            seenObs += [tmpOb]
                        # print(seenObs)

                    tarPos = self.finalTarget.copy()
                    if(g_RelativeCoordinates):
                        tarPos[0] = round(tarPos[0]-self.x, g_RoundPlace)
                        tarPos[1] = round(tarPos[1]-self.y, g_RoundPlace)
                    if(g_SelfDirection):
                        if(not g_RelativeCoordinates):
                            tarPos[0] = round(tarPos[0]-self.x, g_RoundPlace)
                            tarPos[1] = round(tarPos[1]-self.y, g_RoundPlace)
                        tarPos[0],tarPos[1] = rotate_coordinate(tarPos[0], tarPos[1], self.theta)
                    tarPos = F"({round(tarPos[0], g_RoundPlace)},{round(tarPos[1], g_RoundPlace)})"

                    tmpFindNextCheckpoint = findNextCheckpoint
                    if(g_RelativeCoordinates):
                        curPos = "(0,0)"
                    else:
                        curPos = F"({round(self.x, g_RoundPlace)},{round(self.y, g_RoundPlace)})"
                    
                    EnvironmentStatus = ','.join('({:.{}f}, {:.{}f})'.format(point[0],g_RoundPlace, point[1],g_RoundPlace) for point in seenObs)

                    # 替换字段
                    tmpFindNextCheckpoint = tmpFindNextCheckpoint.replace("CURPOS",curPos)   #当前位置坐标
                    tmpFindNextCheckpoint = tmpFindNextCheckpoint.replace("TARPOS",tarPos)   #目标位置坐标
                    tmpFindNextCheckpoint = tmpFindNextCheckpoint.replace("ENVIRONMENTSTATUS",EnvironmentStatus) #障碍物位置坐标
                    # print(tmpFindNextCheckpoint)
                    # print(EnvironmentStatus)

                    print(tmpFindNextCheckpoint)
                    response = ask(tmpFindNextCheckpoint)
                    print(response)
                    # print(type(response))
                    # multi_pos = extract_floats_from_parentheses(response)
                    # print(multi_pos)
                    # x_set = multi_pos[0]
                    # y_set = multi_pos[1]
                    # print(F"Dest:{multi_pos}")

                    pos_x,pos_y = find_two_floats(response)
                    x_set = pos_x
                    y_set = pos_y
                    if(g_SelfDirection):
                        x_set,y_set = inverse_rotate_coordinate(x_set, y_set, self.theta)
                        if(not g_RelativeCoordinates):
                            x_set = round(x_set+self.x, g_RoundPlace)
                            y_set = round(y_set+self.y, g_RoundPlace)
                    if(g_RelativeCoordinates):
                        x_set = round(x_set+self.x, g_RoundPlace)
                        y_set = round(y_set+self.y, g_RoundPlace)

                    self.planning_target=[float(x_set),float(y_set)]
                    if (np.sqrt((self.planning_target[0]-self.finalTarget[0]) ** 2 + (self.planning_target[1]-self.finalTarget[1]) ** 2) < 0.5*g_TargetRadius):
                        self.vplanner.passFlag = False
                    px, py = planner.planning(self.planning_obs[:, 0], self.planning_obs[:, 1], Playground.planning_obs_radius + self.dwaconfig.robot_radius * self.dwaconfig.safety_ratio, self.x, self.y, self.planning_target[0], self.planning_target[1], -10, -10, 10, 10)
                    self.planning_path = np.vstack([px, py]).T
                    self.vplanner_midpos_index = None
                    self.checkpoints = np.append(self.checkpoints, [[self.planning_target[0],self.planning_target[1]]], axis=0)

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

                if (np.sqrt((self.x-self.finalTarget[0]) ** 2 + (self.y-self.finalTarget[1]) ** 2) < g_TargetRadius):
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
            
            if (self.chasingFlag and self.planning_target is not None):
                # print(F"Chasing {self.planning_target}...") # 当前正在追踪哪个点
                if(np.sqrt((self.x-self.planning_target[0]) ** 2 + (self.y-self.planning_target[1]) ** 2) < g_CheckPointRadius):
                    self.arrivedCheckpoint = True

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

        # if self.planning_target is not None:
        #     # self.ax.plot(self.planning_target[0], self.planning_target[1], "rx", markersize=20)
        #     self.ax.plot(self.planning_target[0], self.planning_target[1], "ro", markersize=5)

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
                self.finalTarget = np.array([event.xdata, event.ydata])
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
            if self.finalTarget is not None and self.planner is not None:
                print("do planning...")
                # self.finalTarget = self.planning_target.copy()
                self.chasingFlag = True
                # Offline Version
                # print(self.planning_obs.shape)
                # px, py = planner.planning(self.planning_obs[:, 0], self.planning_obs[:, 1], Playground.planning_obs_radius+ self.dwaconfig.robot_radius * self.dwaconfig.safety_ratio, self.x, self.y, self.planning_target[0], self.planning_target[1], -10, -10, 10, 10)
                # self.planning_path = np.vstack([px, py]).T
                # print("pathLength : ", self.planning_path.shape[0])

                # ids = split_into_parts(self.planning_path.shape[0], 5)
                # for id in ids:
                #     self.temp_targetPos = np.append(self.temp_targetPos, [[self.planning_path[id][0], self.planning_path[id][1]]], axis=0)


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
