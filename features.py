import numpy as np
from scipy.stats import mode

class FeatureConverter():

    def __init__(self, max_stack, possible_actions):
        self.possible_actions = possible_actions
        self.actions = np.linspace(0, len(self.possible_actions)-1, len(self.possible_actions))
        self.max_stack = max_stack
        self.init_feat = 28
        self.features = self.init_feat*len(self.possible_actions)
        self.theta = np.random.randn(self.features)

        pass

    def mode_func(self, y):
        max_y = []
        for i_y in y:
            max_y.append(int(i_y))
        else:
            max_y = 0
        #print(mode(max_y)[0] == [0])
        mode_y = mode(max_y)[0]
        mode_val_y = 0
        for i_y in y:
            if int(i_y) == mode_y:
                mode_val_y += 1
        return mode_y, mode_val_y

    def x_height(self, x, sections):
        x_check = np.linspace(160, 480, sections)
        x_list = []
        for i, checker in enumerate(x_check):
            x_list.append(0)
            for i_x in x:
                if checker + 5 > i_x > checker - 5:
                    x_list[i] += 1
        return x_list

    def x_adj(self, x_list):
        x_adj_list = []
        for i, x_val in enumerate(x_list[1:-2]):
            i+=1
            x_adj_list.append((x_list[i-1] + x_list[i+1]))
        return x_adj_list

    def x_wobble(self, x, sections):
        x_split = []
        x_check = np.linspace(160, 480, sections)
        for i, checker in enumerate(x_check):
            first_loop_right = True
            first_loop_left = True
            wobble_right = 0
            wobble_left = 0
            sect_list = []
            for i_x in x:
                if checker + 5 > i_x > checker - 5:
                    sect_list.append(i_x)
            if len(sect_list) > 0:
                sect_mean = np.mean(sect_list)
                for i_x in sect_list:
                    if i_x > sect_mean:
                        if first_loop_right:
                            wobble_right = i_x - sect_mean
                            first_loop_right = False
                        else:
                            wobble_right = (wobble_right + i_x-sect_mean)/2
                    elif i_x < sect_mean:
                        if first_loop_left:
                            wobble_left = i_x - sect_mean
                            first_loop_right = False
                        else:
                            wobble_left = (wobble_left + i_x-sect_mean)/2
                if abs(wobble_right) > abs(wobble_left):
                    x_split.append(wobble_right)
                else:
                    x_split.append(wobble_left)
            else:
                x_split.append(0)
        return x_split







    def sa2x(self, s, a, t):
        return_array = np.zeros(self.features)
        move_idx = self.possible_actions.tolist().index(a)#
        y = []
        x = []
        for i in s:
            y.append(i[1])
            x.append(i[0])
        max_h = max(y)
        mode_y, mode_val_y = self.mode_func(y)
        x_list = self.x_height(x, 10)
        x_adj_list = self.x_adj(x_list)
        x_wobble_list = self.x_wobble(x, 10)
        f = 0 + move_idx * self.init_feat
        #return_array[f] = np.mean(x) - x[-1]
        #return_array[f+1] = np.mean(x)
        #return_array[f+2] = x[-1] - x[0]
        #return_array[f+3] = t
        #return_array[f+4] = max_h
        #return_array[f+5] = 1
        #return_array[f+6] = np.mean(x) - a # 6
        #return_array[f+7] = mode_y
        #return_array[f+8] = mode_val_y
        #return_array[f+9] = 1
        adder = -1
        for i in x_list:
            adder += 1
            return_array[f+adder] = i

        for i in x_adj_list:
            adder += 1
            return_array[f+adder] = i

        for i in x_wobble_list:
            adder += 1
            return_array[f + adder] = i

        #print(x)
        #print(y)
        return return_array

    def predict(self, s, a, t):
        x = self.sa2x(s, a, t)
        #print(self.theta)
        #print(x)
        #return self.theta.dot(x)
        return np.cos(np.pi * self.theta.dot(x))


    def grad(self, s, a, t):
        return self.sa2x(s, a, t)