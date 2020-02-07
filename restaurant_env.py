import numpy as np


class RestaurantEnv:

    def __init__(self, day_length):
        self.day_length = day_length
        self.fill_prob = 0.2
        self.curr_step = 0
        self.env = np.full([5, 5], 'G')

        self.robot_pos = [0, 0]
        self.charge = 25

        self.food_collect = [0, 2]
        self.charging = [0, 0]
        self.tableA = [2, 0]
        self.tableB = [4, 0]
        self.tableC = [2, 4]
        self.tableD = [4, 4]
        self.fill_tile(self.food_collect, 'F')
        self.fill_tile(self.charging, 'R')
        self.fill_tile(self.tableA, 'A')
        self.fill_tile(self.tableB, 'B')
        self.fill_tile(self.tableC, 'C')
        self.fill_tile(self.tableD, 'D')
        self.table_info_dict = {'A': 'empty', 'B': 'empty', 'C': 'empty', 'D': 'empty'}
        self.food_info_dict = {'A': 'none', 'B': 'none', 'C': 'none', 'D': 'none'}
        self.food_carrying = []
        self.total_served = 0

    def fill_tile(self, pos, category):
        self.env[pos[0], pos[1]] = category

    def update_step(self):
        for key in self.table_info_dict:
            if np.random.uniform(0, 1) < self.fill_prob and self.table_info_dict[key] == 'empty':
                self.table_info_dict[key] = 'waiting_order'
            if np.random.uniform(0, 1) < self.fill_prob and self.table_info_dict[key] == 'eating':
                self.table_info_dict[key] = 'waiting_bill'

            if np.random.uniform(0, 1) < self.fill_prob and self.food_info_dict[key] == 'preparing':
                self.food_info_dict[key] = 'ready'

    def perform_action(self, action):
        self.update_step()

        if self.curr_step == self.day_length:
            return self.total_served/2, 'reset'

        self.charge -= 1
        if self.charge == 0:
            return -100, 'reset'

        elif action == 'move_U':
            if self.robot_pos[0] == 0:
                return -1, 'fail'
            else:
                self.robot_pos[0] -= 1
                return 0, 'success'

        elif action == 'move_D':
            if self.robot_pos[0] == np.shape(self.env)[0]-1:
                return -1, 'fail'
            else:
                self.robot_pos[0] += 1
                return 0, 'success'

        elif action == 'move_L':
            if self.robot_pos[1] == 0:
                return -1, 'fail'
            else:
                self.robot_pos[1] -= 1
                return 0, 'success'

        elif action == 'move_R':
            if self.robot_pos[1] == np.shape(self.env)[0]-1:
                return -1, 'fail'
            else:
                self.robot_pos[1] += 1
                return 0, 'success'

        elif action == 'get_order':
            pos_letter = self.env[self.robot_pos[0], self.robot_pos[1]]
            if pos_letter in self.table_info_dict:
                if self.table_info_dict[pos_letter] == 'waiting_order':
                    self.table_info_dict[pos_letter] = 'waiting_food'
                    self.food_info_dict[pos_letter] = 'preparing'
                    return 0, 'success'
            return 0, 'fail'

        elif action == 'collect_food':
            return_str = 'fail'
            pos_letter = self.env[self.robot_pos[0], self.robot_pos[1]]
            if pos_letter == 'F':
                for key in self.food_info_dict:
                    if self.food_info_dict[key] == 'ready':
                        self.food_info_dict[key] = 'none'
                        self.food_carrying.append(key)
                        return_str = 'success'
            return 0, return_str

        elif action == 'serve_food':
            pos_letter = self.env[self.robot_pos[0], self.robot_pos[1]]
            if pos_letter in self.food_carrying:
                self.food_carrying.pop(self.food_carrying.index(pos_letter))
                self.table_info_dict[pos_letter] = 'eating'
                self.total_served += 1
                return 0, 'success'
            return 0, 'fail'

        elif action == 'give_bill':
            pos_letter = self.env[self.robot_pos[0], self.robot_pos[1]]
            if pos_letter in self.table_info_dict:
                self.table_info_dict[pos_letter] = 'empty'
                return 1, 'success'
            return 0, 'fail'

        elif action == 'charging':
            if self.robot_pos == self.charging:
                self.charge = 25
                return 0, 'success'
            else:
                return 0, 'fail'

    def reset(self):
        self.table_info_dict = {'A': 'empty', 'B': 'empty', 'C': 'empty', 'D': 'empty'}
        self.curr_step = 0
        self.total_served = 0


if __name__ == '__main__':
    env = RestaurantEnv(10)
    print(env.env)
    thingy = 'none'
    total_reward = 0
    while thingy != 'reset':
        inputter = input()
        reward, thingy = env.perform_action(inputter)
        print(thingy)
        print(env.food_carrying)
        print(env.robot_pos)
        print(env.charge)
        print(env.food_info_dict)
        print(env.table_info_dict)
        total_reward += reward
        print(total_reward)
