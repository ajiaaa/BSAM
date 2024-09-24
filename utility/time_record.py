import time

class TIME_RECORD:
    def __init__(self):
        self.time_dict = {}

    def record(self, flag, witch_time):
        key = self.time_dict.keys()
        if 'whole_' + witch_time not in key:
            self.time_dict['whole_' + witch_time] = 0
        if flag == 'start':
            self.time_dict['s_' + witch_time] = time.time()
        if flag == 'end':
            self.time_dict['e_' + witch_time] = time.time()
            self.time_dict['whole_' + witch_time] += self.time_dict['e_' + witch_time] - self.time_dict['s_' + witch_time]
            del self.time_dict['s_' + witch_time]
            del self.time_dict['e_' + witch_time]
        return self.time_dict
