from utils import *
import pickle
import pandas as pd

path = 'data/strokeData/'
headers = pd.read_csv('data/strokeData/field_v3.txt')
subject = 'ST05'
assistance_levels = ['L0R0', 'L0R1', 'L1R1', 'L2R1', 'L2R2', 'L3R1']
window_size = 80

# Window All Data
for level in assistance_levels:
    data = pd.read_csv(path+subject+'/labeled_' + level + '.txt', index_col=0)
    data['leftGaitPhase'] = convert_to_gp(data['leftGaitPhaseX'], data['leftGaitPhaseY'])
    data['rightGaitPhase'] = convert_to_gp(data['rightGaitPhaseX'], data['rightGaitPhaseY'])

    print(data.shape)
    windowed_dataset = window_data(data, window_size, ['leftGaitPhaseX', 'leftGaitPhaseY', 'rightGaitPhaseX', 'rightGaitPhaseY', 'leftGaitPhase', 'rightGaitPhase'])

    export_filename = path + subject + '/windowed_' + level + '.pkl'
    with open(export_filename, 'wb') as f:
        pickle.dump(windowed_dataset, f)
        f.close()