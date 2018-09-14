class Config:
    """create a config instance, do some settings for this project"""
    def __init__(self, cfg):
        """
        cfg: config file path
        """
        self.init(cfg)

    def init(self, cfg):
        """
        Parameters:
            cfg : string; config file path
        """
        self.cfgdic = {}

        with open(cfg, 'r') as f:
            for aline in f.readlines():
                if aline != '\n' and aline[0] != '*' and aline[0] != ' ':   
                    self.cfgdic[aline.strip().split()[0]] = aline.strip().split()[-1]

        # # check if has setted those three basic parameters in config file
        # if not self.cfgdic.get('frame_size', None) or\
        #    not self.cfgdic.get('frame_shift', None) or\
        #    not self.cfgdic.get('segment_size', None) or\
        #    not self.cfgdic.get('segment_shift', None):
        #     raise ValueError('Please set basic parameters in config file: frame_size, frame_shift, segment_size, segment_shift')
        # else:
        #     self.cfgdic['frame_size'] = float(self.cfgdic['frame_size'])
        #     self.cfgdic['frame_shift'] = float(self.cfgdic['frame_shift'])
        #     self.cfgdic['segment_size'] = float(self.cfgdic['segment_size'])
        #     self.cfgdic['segment_shift'] = float(self.cfgdic['segment_shift'])

        # # set the system's operating sample rate, default is 16000
        # if not self.cfgdic.get('operating_rate', None):
        #     self.cfgdic['operating_rate'] = 16000
        # else:
        #     self.cfgdic['operating_rate'] = int(self.cfgdic['operating_rate'])

        # if not self.cfgdic.get('mfcc_order', None):
        #     raise ValueError('Please set the order of mfcc in config file!')
        # else:
        #     self.cfgdic['mfcc_order'] = int(self.cfgdic['mfcc_order'])

        # if not self.cfgdic.get('roll_percent', None):
        #     raise ValueError('Please set the percent of roll off frequence in config file!')
        # else:
        #     self.cfgdic['roll_percent'] = float(self.cfgdic['roll_percent'])

        # if not self.cfgdic.get('n_fft', None):
        #     raise ValueError('Please set the number of fft points in config file!')
        # else:
        #     self.cfgdic['n_fft'] = int(self.cfgdic['n_fft'])