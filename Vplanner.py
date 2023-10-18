import numpy as np

class DWA():
    def __init__(self):
        self.passFlag = True
        pass

    def plan(self, x, info, midpos, planning_obs):  # x = [self.x,self.y,self.theta,self.vx,self.vw]
        vw = self.vw_generate(x, info)  # 更新当前可以变化的最大/小线速度，最大/小角速度
        min_score = 1000.0
        # 速度v,w都被限制在速度空间里
        all_ctral = []
        all_scores = []
        u = np.array([vw[0], vw[2]])
        for v in np.arange(vw[0], vw[1], info.v_reso):  # 遍历每个线速度和角速度
            for w in np.arange(vw[2], vw[3], info.yawrate_reso):
                # cauculate traj for each given (v,w)
                ctraj = self.traj_cauculate(x, [v, w], info)
                # 计算评价函数
                goal_score = info.to_goal_cost_gain * self.goal_evaluate(ctraj, midpos, x, passFlag = self.passFlag)
                vel_score = info.speed_cost_gain * self.velocity_evaluate(ctraj, info)
                traj_score = info.obstacle_cost_gain * self.traj_evaluate(ctraj, planning_obs, info)
                    
                # 可行路径不止一条，通过评价函数确定最佳路径
                # 路径总分数 = 距离目标点 + 速度 + 障碍物
                # 分数越低，路径越优
                ctraj_score = goal_score + vel_score + traj_score
                # print(goal_score, vel_score, traj_score)

                ctraj = np.reshape(ctraj, (-1, 5))
                # print(ctraj)
                # print(ctraj.shape)
                # evaluate current traj (the score smaller,the traj better)
                if min_score >= ctraj_score:
                    min_score = ctraj_score
                    u = np.array([v, w])  # 更新u存储最好的线速度v和角速度w
                    best_ctral = ctraj
                all_ctral.append(ctraj)
                all_scores.append(ctraj_score)

        return u, best_ctral, all_ctral, all_scores

    # 定义机器人运动模型
    # 返回坐标(x,y),偏移角theta,速度v,角速度w
    def motion_model(self, x, u, dt):
        # robot motion model: x,y,theta,v,w
        x[0] += u[0] * dt * np.cos(x[2])
        x[1] += u[0] * dt * np.sin(x[2])
        x[2] += u[1] * dt
        x[3] = u[0]
        x[4] = u[1]
        return x

    # 依据当前位置及速度，预测轨迹 u为某一线速度和角速度
    def traj_cauculate(self, x, u, info):
        ctraj = np.array(x)
        xnew = np.array(x)
        time = 0

        while time <= info.predict_time:  # 在preditc_time内，该输入到达的轨迹
            xnew = self.motion_model(xnew, u, info.dt)
            ctraj = np.vstack([ctraj, xnew])
            time += info.dt

        # print(ctraj)
        return ctraj
        
    # 产生速度空间  # x = [self.x,self.y,self.theta,self.vx,self.vw]
    def vw_generate(self, x, info):
        # generate v,w window for traj prediction
        Vinfo = [info.min_speed, info.max_speed,
                 info.min_yawrate, info.max_yawrate]

        Vmove = [x[3] - info.max_accel * info.dt,
                 x[3] + info.max_accel * info.dt,
                 x[4] - info.max_dyawrate * info.dt,
                 x[4] + info.max_dyawrate * info.dt]

        # 保证速度变化不超过info限制的范围
        vw = [max(Vinfo[0], Vmove[0]), min(Vinfo[1], Vmove[1]),
              max(Vinfo[2], Vmove[2]), min(Vinfo[3], Vmove[3])]

        return vw

    # 距离目标点评价函数
    def goal_evaluate(self, traj, goal, x, passFlag = False):
        # cauculate current pose to goal with euclidean distance  求轨迹最后一个点至终点的欧氏距离
        if (passFlag == True or (passFlag == False and (np.sqrt((x[0]-goal[0]) ** 2 + (x[1]-goal[1]) ** 2)) > 2.5)):
            minDistScore = 99999
            minid = 99999
            count = 0 
            for trajPos in traj:
                count +=1

                goal_score = np.sqrt((trajPos[0]-goal[0]) ** 2 + (trajPos[1]-goal[1]) ** 2)
                if goal_score < minDistScore:
                    minid = count
                    minDistScore = min(minDistScore, goal_score)
            return minDistScore +0.01*minid
        else:
            goal_score = np.sqrt((traj[-1, 0]-goal[0]) ** 2 + (traj[-1, 1]-goal[1]) ** 2)
            return goal_score

    # 速度评价函数
    def velocity_evaluate(self, traj, info):
        # cauculate current velocty score
        vel_score = info.max_speed - traj[-1, 3]
        return vel_score

    # 轨迹距离障碍物的评价函数
    def traj_evaluate(self, traj, obstacles, info):
        # evaluate current traj with the min distance to obstacles
        min_dis = float("Inf")
        for i in range(len(traj)):
            for ii in range(len(obstacles)): # 遍历每个障碍
                current_dist = np.sqrt((traj[i, 0] - obstacles[ii, 0])**2 + (traj[i, 1] - obstacles[ii, 1])**2)

                if current_dist <= (info.obs_radius + info.robot_radius * info.safety_ratio):
                    return float("Inf")

                if min_dis >= current_dist:
                    min_dis = current_dist

        return 1 / min_dis  # 距离障碍越近，分数越大
