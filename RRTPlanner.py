from collections import namedtuple
from random import randint, uniform
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

Point = namedtuple('Point', 'x y')
Node = namedtuple('Node', 'id pos parent')

class RRTPlanner:
    def __init__(self, unit_step = 0.1):
        self.unit_step = unit_step
        self.GOAL_CHANCE = 0.2
        self.velocity = 0.2
        self.planning_obs_radius = 0.5
        
    def planning(self, planning_obs_x, planning_obs_y, planning_obs_radius, planning_start_x, planning_start_y, planning_target_x, planning_target_y, planning_minx, planning_miny, planning_maxx, planning_maxy):
        # 离散化
        planning_start_x_int = int((planning_start_x - planning_minx) / self.unit_step)
        planning_start_y_int = int((planning_start_y - planning_miny) / self.unit_step)
        planning_target_x_int = int((planning_target_x - planning_minx) / self.unit_step)
        planning_target_y_int = int((planning_target_y - planning_miny) / self.unit_step)
        self.planning_max_x_int = int((planning_maxx - planning_minx) / self.unit_step)
        self.planning_max_y_int = int((planning_maxy - planning_miny) / self.unit_step)
        
        self.planning_minx = planning_minx
        self.planning_miny = planning_miny
        self.planning_maxx = planning_maxx
        self.planning_maxy = planning_maxy
        # 设置障碍物
        obstacles = []
        planning_obs_radius *= 1.0
        self.obs_radius_int = int(planning_obs_radius / self.unit_step)
        for i in range(planning_obs_x.shape[0]):
            obs_x_int = int((planning_obs_x[i] - planning_minx) / self.unit_step)
            obs_y_int = int((planning_obs_y[i] - planning_miny) / self.unit_step)
            obstacles.append(Point(obs_x_int,obs_y_int))
        # print(obstacles)
        start_pos = Point(planning_start_x_int, planning_start_y_int)
        end_region = Point(planning_target_x_int, planning_target_y_int)
        nodes = self.calculate_path(start_pos, end_region, obstacles)
        node_count = len(nodes)
        
        result = []
        current_node = nodes[-1]
        while current_node.id != 0:
            parent = nodes[current_node.parent]
            result.insert(0,[current_node.pos.x * self.unit_step + planning_minx, current_node.pos.y * self.unit_step + planning_minx]) #, parent.pos.x, parent.pos.y
            current_node = parent

        result = np.array(result)
        print("Nodes Calculated {}".format(node_count))
        return result[:,0],result[:,1]

    def dist(self, point1, point2):
        return sqrt((point1[0] - point2[0]) ** 2 + (point1.y - point2.y) ** 2)

    def in_regions(self,point, regionsPoint):
        for regionPoint in regionsPoint:
            if (self.dist(point, regionPoint) < self.obs_radius_int ):#+ safety distance
                return True
        return False

    def get_closest(self, nodes, point):  # Return the node in the list that's closest to the given point
        return min(nodes, key=lambda x: self.dist(x.pos, point))

    def steer(self,point1, point2):  # Return an intermediate point on the line between point1 and point2
        total_offset = abs(point2.x - point1.x) + abs(point2.y - point1.y)
        x = point1.x + self.velocity * ((point2.x - point1.x) / total_offset)
        y = point1.y + self.velocity * ((point2.y - point1.y) / total_offset)
        return Point(x, y)

    def calculate_path(self,start, goal, obstacles):
        nodes = [Node(0, start, 0)]
        while True:
            if uniform(0, 1) < self.GOAL_CHANCE:
                z_rand = Point(goal.x , goal.y)
            else:
                z_rand = Point(randint(0,self.planning_max_x_int), randint(0,self.planning_max_y_int))
            # print(obstacles)
            if self.in_regions(z_rand, obstacles):
                continue
            nearest = self.get_closest(nodes, z_rand)
            if z_rand == nearest.pos:
                continue

            new_pos = self.steer(nearest.pos, z_rand)
            if self.in_regions(new_pos, obstacles):
                continue
            nodes.append(Node(len(nodes), new_pos, nearest.id))
            if len(nodes) % 100 == 0:
                print("{} Nodes Searched".format(len(nodes)))
            if self.dist(new_pos, goal) < (self.obs_radius_int/2):  # goal region
                return nodes
