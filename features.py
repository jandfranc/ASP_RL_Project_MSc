import numpy as np

class FeatureConverter():

    def __init__(self, max_stack, possible_actions):
        self.possible_actions = possible_actions
        self.actions = np.linspace(0, len(self.possible_actions)-1, len(self.possible_actions))
        self.max_stack = max_stack
        self.init_feat = 6
        self.features = self.init_feat*len(self.possible_actions)
        self.theta = np.ones(self.features)
        self.theta = self.theta / np.sum(self.theta)
        pass

    '''
    def sa2x(self, s, a, t):
        return_array = np.zeros(self.features)
        f = 0
        for i in range(t):
            if i == t-1:
                return_array[f] = s[i] * a
                return_array[f+1] = s[i]**2 / a
                return_array[f+2] = s[i]**3 /a
                f += 3
            else:
                return_array[f] = s[i] * a
                return_array[f + 1] = s[i] / a
                return_array[f + 2] = s[i] + a
                return_array[f + 3] = abs(s[i] - s[i+1]) / a
                return_array[f + 4] = abs(s[i] - s[i+1]) * a
                return_array[f + 5] = abs(s[i] - s[i+1]) * a ** 2
                f += 6
        return return_array
    '''

    def sa2x(self, s, a, t):
        return_array = np.zeros(self.features)
        move_idx = self.possible_actions.tolist().index(a)
        y = []
        x = []
        for i in s:
            y.append(i[1])
            x.append(i[0])
        max_h = max(y)
        f = 0 + move_idx * self.init_feat
        return_array[f] = np.mean(x) - s[-1][0]
        return_array[f+1] = np.mean(x)
        return_array[f+2] = s[-1][0] - s[0][0]
        return_array[f+3] = t
        return_array[f+4] = max_h
        return_array[f+5] = 1


        return return_array

    def predict(self, s, a, t):
        x = self.sa2x(s, a, t)
        #print(self.theta)
        #print(x)
        return self.theta.dot(x)


    def grad(self, s, a, t):
        return self.sa2x(s, a, t)