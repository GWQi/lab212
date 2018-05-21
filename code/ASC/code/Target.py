from Learn import Power
import zipfile
import yaml
import pickle


class Test(object):
    """
    this class is to perform segmentaion of a given audio signal into speech and music.

    parameters:
        system : string; system setting zip file path
        cfg : string; config file path

    attributes:
        es_ranking_list : list of Power instances, ranked by extrem speech threshold separation power
        em_ranking_list : list of Power instances, ranked by extrem music threshold separation power
        hs_ranking_list : list of Power instances, ranked by high probability speech threshold separation power
        hm_ranking_list : list of Power instances, ranked by high probability music threshold separation power
        sp_ranking_list : list of Power instances, ranked by speech, music separation threshold separation power
        setting : dict, dict of system setting
        config : dict, dict of target configeration
        with_statistics : bool, segmented by extracted feature statistics
        batch : 
    """
    def __init__(self, system, cfg):

        # ***************************system setting configuration******************************
        try:
            with zipfile.ZipFile(system) as myzip:
                # load extrem speech threshold separation power ranking list
                with myzip.open('es_ranking_list', 'r') as f:
                    self.es_ranking_list = pickle.load(f)

                # load extrem music threshold separation power ranking list
                with myzip.open('em_ranking_list', 'r') as f:
                    self.em_ranking_list = pickle.load(f)

                # load high probability speech threshold separation power ranking list
                with myzip.open('hs_ranking_list', 'r') as f:
                    self.hs_ranking_list = pickle.load(f)

                # load high probability music threshold separation power ranking list
                with myzip.open('hm_ranking_list', 'r') as f:
                    self.hm_ranking_list = pickle.load(f)

                # load speech, music separation threshold separation power ranking list
                with myzip.open('sp_ranking_list', 'r') as f:
                    self.sp_ranking_list = pickle.load(f)

                # load system setting dict
                with myzip.open('setting.yaml', 'r') as f:
                    self.setting = yaml.load(f)
        except:
            raise ValueError("system setting file is damaged, please check: {}".format(system))

        # target config
        try:
            with open(cfg, 'r') as f:
                self.config = yaml.load(f)
        except:
            raise ValueError("Loading Target config file failed, please check: {}".format(cfg))

        # *****************************check target config*******************************
        # check log file path
        if self.config.get('log_file_path', None) == None:
            raise ValueError("Please set log file path in cfg file: {}".format(cfg))

        # check source data files path
        if self.config.get('source_data_path', None) == None:
            raise ValueError("Please give a correct data files path in cfg file: {}".format(cfg))








def main(argv):
    try:
        opts, args = getopt.getopt(argv, 'hc:s:',['help', 'config=', 'sys='])
    except getopt.GetoptError as e:
        print('python Target.py -s <system setting file path>-c <config file path>')

    for opt, value in opts:
        if opt in ['-h', '--help']:
            print('python Target.py -s <system setting file path>-c <config file path>')
        elif opt in ['-s', '--sys']:
            system = value
        elif opt in ['-c', '--config']:
            cfg = value
    try:
        Test(system, cfg)
    except:
        print('python Target.py -s <system setting file path>-c <config file path>')

if __name__ == '__main__':
    main(sys.argv[1:])