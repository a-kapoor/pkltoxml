import sys
import pickle

file_name=sys.argv[1]

bdt=pickle.load(open(file_name, 'rb'))

import ct as ct

def run(pklFile):
    xml_file = pklFile.replace('.pkl', '.xml')
    try:
        with open(pklFile, 'rb') as pklOpen:
            pkl_data = pickle.load(pklOpen)
            print('pklData loaded')
    except IOError:
        print('IOError when loading pklData from the file')
    bst = pkl_data.best_estimator_.get_booster()
    features = bst.feature_names
    print(features)
    bdt_model = ct.BDTxgboost(pkl_data.best_estimator_, features, ['Background', 'Signal'])
    bdt_model.to_tmva(xml_file)
    print('.xml BDT model saved to ' + str(xml_file))

run('/home/akapoor/ID-Trainer/TTHTagger/XGB/XGB_modelXGB.pkl')


