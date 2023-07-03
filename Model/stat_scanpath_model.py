import numpy as np


class satisticalScanPath():
    def __init__(
        self,
    ):
        self.name    = 'statistical scan path model'
        
    def sample_fix_duraton_prob(self, word):
        word_len = len(word)
        if word_len == 1:
            prob = 0.077
            duration = 209
        elif word_len == 2:
            prob = 0.205
            duration = 215
        elif word_len == 3:
            prob = 0.318
            duration = 210
        elif word_len == 4:
            prob = 0.48
            duration = 205
        elif word_len == 5:
            prob = 0.8
            duration = 229
        elif word_len == 6:
            prob = 0.825
            duration = 244
        elif word_len == 7:
            prob = 0.875
            duration = 258
        elif word_len == 8:
            prob = 0.915
            duration = 260
        elif word_len >= 9:
            prob = 0.94
            duration = 276
        return (prob, duration)

    # input:
    #   text: list of triplets (word,x_pos,y_pos) for all the words in the text
    def sample_postions_for_text(self, text,
                                screen_config = None):
        if screen_config is None:
            screen_config = {  'px_x':1680,
                  'px_y':1050,
                  'max_dva_x': 30,
                  'max_dva_y': 25
                 }
        x_locations = []
        y_locations = []
        fix_durations = []
        for i in range(len(text)):
            word,x_pos,y_pos = text[i]
            prob, duration = self.sample_fix_duraton_prob(word)
            # choice
            choice = np.random.choice([0,1],size=(1,),p=[1-prob,prob])[0]
            if choice == 1:
                x_locations.append(x_pos)
                y_locations.append(y_pos)
                fix_durations.append(duration)
        x_locations = np.array(x_locations)
        y_locations = np.array(y_locations)
        fix_durations = np.array(fix_durations)
        x_dva = x_locations / screen_config['px_x'] * screen_config['max_dva_x']
        y_dva = y_locations / screen_config['px_y'] * screen_config['max_dva_y']
        return x_locations, y_locations, x_dva, y_dva, fix_durations
    
    
    def dva_to_vel(self, vector):
        vel = np.array(vector[1:]) - np.array(vector[0:-1])
        vel = np.array([0] + list(vel))
        return vel
