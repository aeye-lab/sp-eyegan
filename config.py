# path to GazeBase data
# download via https://figshare.com/articles/dataset/GazeBase_Data_Repository/12912257
from __future__ import annotations
GAZE_BASE_DIR = 'data/GazeBase/raw/'

# path to SB-SAT raw data -- published upon acceptance
SB_SAT_DIR_PATH = 'data/SB-SAT'

# specify where you want to save your contrastively pre-trained models
CONTRASTIVE_PRETRAINED_MODELS_DIR = 'pretrain_model/'#'/path/where/to/save/models/'

# specify where you want to save your downstream models
TRAINED_CLASSIFICATION_MODELS_DIR = 'model/'#'/path/where/to/save/models/'

# specify where you want to save your results
CSV_RESULTS_FILE = 'results/subj_results.csv'

# biometrics data dir
JUDO_BASE_DIR = 'data/Judo1000/'

# ADHD data
ADHD_DATA_DIR = 'data/ecml-ADHD/Data/X_px/'
ADHD_LABEL_PATH = 'data/ecml-ADHD/Data/sub_info/sub_sel_classif.csv'

# Gaze on Faces data
GOF_DATA_DIR = 'data/Gaze_on_Faces/gaze_csv/'
GOF_INFO_PATH = 'data/Gaze_on_Faces/observer_info'
