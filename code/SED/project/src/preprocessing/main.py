import sys
sys.path.append("/home/gwq/git/lab212/code/")
from dcase_preprocess import DCASEData2017Task2
dcase2017task3 = DCASEData2017Task2("/home/gwq/dataset/dcase2017")
dcase2017task3.cutFeatures()