import numpy as np
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.model_selection import train_test_split
import sklearn.tree
import sklearn.svm
import sklearn.discriminant_analysis
import pmlb

import classifier
import utils
from optimalsplitboost import OptimalSplitGradientBoostingClassifier

USE_SIMULATED_DATA = True # True
USE_01_LOSS = False # False
TEST_SIZE = 0.20 # .10

########################
## PMLB Dataset sizes ##
########################
if False:
    from pmlb import classification_dataset_names, regression_dataset_names
    for dataset_name in classification_dataset_names:
        X,y = pmlb.fetch_data(dataset_name, return_X_y=True)
        print(dataset_name, X.shape, np.unique(y))

# GAMETES_Epistasis_2_Way_1000atts_0.4H_EDM_1_EDM_1_1 (1600, 1000) [0 1]
# GAMETES_Epistasis_2_Way_20atts_0.1H_EDM_1_1 (1600, 20) [0 1]
# GAMETES_Epistasis_2_Way_20atts_0.4H_EDM_1_1 (1600, 20) [0 1]
# GAMETES_Epistasis_3_Way_20atts_0.2H_EDM_1_1 (1600, 20) [0 1]
# GAMETES_Heterogeneity_20atts_1600_Het_0.4_0.2_50_EDM_2_001 (1600, 20) [0 1]
# GAMETES_Heterogeneity_20atts_1600_Het_0.4_0.2_75_EDM_2_001 (1600, 20) [0 1]
# Hill_Valley_with_noise (1212, 100) [0 1]
# Hill_Valley_without_noise (1212, 100) [0 1]
# adult (48842, 14) [0 1]
# agaricus_lepiota (8145, 22) [0 1]
# allbp (3772, 29) [0 1 2]
# allhyper (3771, 29) [0 1 2 3]
# allhypo (3770, 29) [0 1 2]
# allrep (3772, 29) [0 1 2 3]
# analcatdata_aids (50, 4) [0 1]
# analcatdata_asbestos (83, 3) [0 1]
# analcatdata_authorship (841, 70) [0 1 2 3]
# analcatdata_bankruptcy (50, 6) [0 1]
# analcatdata_boxing1 (120, 3) [0 1]
# analcatdata_boxing2 (132, 3) [0 1]
# analcatdata_creditscore (100, 6) [0 1]
# analcatdata_cyyoung8092 (97, 10) [0 1]
# analcatdata_cyyoung9302 (92, 10) [0 1]
# analcatdata_dmft (797, 4) [0 1 2 3 4 5]
# analcatdata_fraud (42, 11) [0 1]
# analcatdata_germangss (400, 5) [0 1 2 3]
# analcatdata_happiness (60, 3) [0 1 2]
# analcatdata_japansolvent (52, 9) [0 1]
# analcatdata_lawsuit (264, 4) [0 1]
# ann_thyroid (7200, 21) [1 2 3]
# appendicitis (106, 7) [0 1]
# australian (690, 14) [0 1]
# auto (202, 25) [-1  0  1  2  3]
# backache (180, 32) [0 1]
# balance_scale (625, 4) [0 1 2]
# biomed (209, 8) [0 1]
# breast (699, 10) [0 1]
# breast_cancer (286, 9) [0 1]
# breast_cancer_wisconsin (569, 30) [0 1]
# breast_w (699, 9) [0 1]
# buggyCrx (690, 15) [0 1]
# bupa (345, 5) [0 1]
# calendarDOW (399, 32) [1 2 3 4 5]
# car (1728, 6) [0 1 2 3]
# car_evaluation (1728, 21) [0 1 2 3]
# cars (392, 8) [0 1 2]
# chess (3196, 36) [0 1]
# churn (5000, 20) [0 1]
# clean1 (476, 168) [0 1]
# clean2 (6598, 168) [0 1]
# cleve (303, 13) [0 1]
# cleveland (303, 13) [0 1 2 3 4]
# cleveland_nominal (303, 7) [0 1 2 3 4]
# cloud (108, 7) [0 1 2 3]
# cmc (1473, 9) [1 2 3]
# coil2000 (9822, 85) [0 1]
# colic (368, 22) [0 1]
# collins (485, 23) [ 0  1  2  3  4  5  6  7  8  9 10 12 13]
# confidence (72, 3) [0 1 2 3 4 5]
# connect_4 (67557, 42) [0 1 2]
# contraceptive (1473, 9) [1 2 3]
# corral (160, 6) [0 1]
# credit_a (690, 15) [0 1]
# credit_g (1000, 20) [0 1]
# crx (690, 15) [0 1]
# dermatology (366, 34) [1 2 3 4 5 6]
# diabetes (768, 8) [1 2]
# dis (3772, 29) [0 1]
# dna (3186, 180) [1 2 3]
# ecoli (327, 7) [0 1 4 5 7]
# fars (100968, 29) [0 1 2 3 4 5 6 7]
# flags (178, 43) [0 1 2 5 6]
# flare (1066, 10) [0 1]
# german (1000, 20) [0 1]
# glass (205, 9) [1 2 3 5 7]
# glass2 (163, 9) [0 1]
# haberman (306, 3) [1 2]
# hayes_roth (160, 4) [1 2 3]
# heart_c (303, 13) [0 1]
# heart_h (294, 13) [0 1]
# heart_statlog (270, 13) [0 1]
# hepatitis (155, 19) [1 2]
# horse_colic (368, 22) [1 2]
# house_votes_84 (435, 16) [0 1]
# hungarian (294, 13) [0 1]
# hypothyroid (3163, 25) [0 1]
# ionosphere (351, 34) [0 1]
# iris (150, 4) [0 1 2]
# irish (500, 5) [0 1]
# kddcup (494020, 41) [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22]
# kr_vs_kp (3196, 36) [0 1]
# krkopt (28056, 6) [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17]
# labor (57, 16) [0 1]
# led24 (3200, 24) [0 1 2 3 4 5 6 7 8 9]
# led7 (3200, 7) [0 1 2 3 4 5 6 7 8 9]
# letter (20000, 16) [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
#  25 26]
# lupus (87, 3) [0 1]
# lymphography (148, 18) [1 2 3 4]
# magic (19020, 10) [0 1]
# mfeat_factors (2000, 216) [0 1 2 3 4 5 6 7 8 9]
# mfeat_fourier (2000, 76) [0 1 2 3 4 5 6 7 8 9]
# mfeat_karhunen (2000, 64) [0 1 2 3 4 5 6 7 8 9]
# mfeat_morphological (2000, 6) [0 1 2 3 4 5 6 7 8 9]
# mfeat_pixel (2000, 240) [0 1 2 3 4 5 6 7 8 9]
# mfeat_zernike (2000, 47) [0 1 2 3 4 5 6 7 8 9]
# mnist (70000, 784) [0 1 2 3 4 5 6 7 8 9]
# mofn_3_7_10 (1324, 10) [0 1]
# molecular_biology_promoters (106, 57) [0 1]
# monk1 (556, 6) [0 1]
# monk2 (601, 6) [0 1]
# monk3 (554, 6) [0 1]
# movement_libras (360, 90) [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15]
# mushroom (8124, 22) [0 1]
# mux6 (128, 6) [0 1]
# new_thyroid (215, 5) [1 2 3]
# nursery (12958, 8) [0 1 3 4]
# optdigits (5620, 64) [0 1 2 3 4 5 6 7 8 9]
# page_blocks (5473, 10) [1 2 3 4 5]
# parity5 (32, 5) [0 1]
# parity5+5 (1124, 10) [0 1]
# pendigits (10992, 16) [0 1 2 3 4 5 6 7 8 9]
# penguins (333, 7) [0 1 2]
# phoneme (5404, 5) [0 1]
# pima (768, 8) [0 1]
# poker (1025010, 10) [0 1 2 3 4 5 6 7 8 9]
# postoperative_patient_data (88, 8) [0 2]
# prnn_crabs (200, 7) [0 1]
# prnn_fglass (205, 9) [0 2 3 4 5]
# prnn_synth (250, 2) [0 1]
# profb (672, 9) [0 1]
# ring (7400, 20) [0 1]
# saheart (462, 9) [0 1]
# satimage (6435, 36) [1 2 3 4 5 7]
# schizo (340, 14) [0 1 2]
# segmentation (2310, 19) [0 1 2 3 4 5 6]
# shuttle (58000, 9) [1 2 3 4 5 6 7]
# sleep (105908, 13) [0 1 2 3 5]
# solar_flare_1 (315, 12) [0 1 2 3 5]
# solar_flare_2 (1066, 12) [0 1 2 3 4 5]
# sonar (208, 60) [0 1]
# soybean (675, 35) [ 0  1  2  3  4  5  6  7  8  9 10 11 12 14 15 16 17 18]
# spambase (4601, 57) [0 1]
# spect (267, 22) [0 1]
# spectf (349, 44) [0 1]
# splice (3188, 60) [0 1 2]
# tae (151, 5) [1 2 3]
# texture (5500, 40) [ 2  3  4  6  7  8  9 10 12 13 14]
# threeOf9 (512, 9) [0 1]
# tic_tac_toe (958, 9) [0 1]
# tokyo1 (959, 44) [0 1]
# twonorm (7400, 20) [0 1]
# vehicle (846, 18) [1 2 3 4]
# vote (435, 16) [0 1]
# vowel (990, 13) [ 0  1  2  3  4  5  6  7  8  9 10]
# waveform_21 (5000, 21) [0 1 2]
# waveform_40 (5000, 40) [0 1 2]
# wdbc (569, 30) [0 1]
# wine_quality_red (1599, 11) [3 4 5 6 7 8]
# wine_quality_white (4898, 11) [3 4 5 6 7 8 9]
# wine_recognition (178, 13) [1 2 3]
# xd6 (973, 9) [0 1]
# yeast (1479, 8) [0 1 2 3 4 5 6 7 8]

##########################
## Generate Random Data ##
##########################
if (USE_SIMULATED_DATA):
    SEED = 256 # 254
    NUM_SAMPLES = 20000 # 1000, 10000
    NUM_FEATURES = 500 # 20, 500
    rng = np.random.RandomState(SEED)
    
    X,y = make_classification(random_state=SEED, n_samples=NUM_SAMPLES, n_features=NUM_FEATURES)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE)
##############################
## END Generate Random Data ##
##############################
else:
    # Breast cancer data
    # data = load_breast_cancer()
    # X, y = data.data, data.target
    # X,y = pmlb.fetch_data('spambase', return_X_y=True)
    # X,y = pmlb.fetch_data('chess', return_X_y=True)
    # X,y = pmlb.fetch_data('churn', return_X_y=True) # X.shape = (5000, 20)
    # X,y = pmlb.fetch_data('twonorm', return_X_y=True) # X.shape = (7400,20)
    # X,y = pmlb.fetch_data('clean2', return_X_y=True) # X.shape = (6598, 168)
    # X,y = pmlb.fetch_data('kr_vs_kp', return_X_y=True) # X.shape = (3196, 36)
    # X,y = pmlb.fetch_data('phoneme', return_X_y=True) # X.shape = (5404, 5)
    # X,y = pmlb.fetch_data('xd6', return_X_y=True) # X.shape = (973, 9)
    # X,y = pmlb.fetch_data('mushroom', return_X_y=True) # X.shape = (8124, 22)
    # X,y = pmlb.fetch_data('dis', return_X_y=True) # X.shape = (3772, 29)
    # X,y = pmlb.fetch_data('GAMETES_Heterogeneity_20atts_1600_Het_0.4_0.2_50_EDM_2_001', return_X_y=True) # X.shape = (1600, 20)
    # X,y = pmlb.fetch_data('GAMETES_Epistasis_2_Way_1000atts_0.4H_EDM_1_EDM_1_1', return_X_y=True) # X.shape = (1600, 1000)
    X,y = pmlb.fetch_data('flare', return_X_y=True) # X.shape (1066, 10)


# GAMETES_Epistasis_2_Way_1000atts_0.4H_EDM_1_EDM_1_1 (1600, 1000) [0 1]

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=TEST_SIZE)

if USE_01_LOSS:
    X_train = 2*X_train - 1
    X_test = 2*X_test - 1
    y_train = 2*y_train - 1
    y_test = 2*y_test - 1
#############################
## Generate Empirical Data ##
#############################
if __name__ == '__main__':

    num_steps = 300

    distiller = classifier.classifierFactory(sklearn.tree.DecisionTreeClassifier) # use classifier
    # print('USING LDA')
    # distiller = classifier.classifierFactory(sklearn.discriminant_analysis.LinearDiscriminantAnalysis)
    # distiller = classifier.classifierFactory(sklearn.tree.DecisionTreeRegressor)

    # From pmlb driver
    # clfKwargs = { 'min_partition_size':            min_partition_size,
    #               'max_partition_size':            max_partition_size,
    #               'row_sample_ratio':              row_sample_ratio,
    #               'col_sample_ratio':              col_sample_ratio,
    #               'gamma':                         gamma,
    #               'eta':                           eta,
    #               'num_classifiers':               num_steps,
    #               'use_constant_term':             False,
    #               'solver_type':                   'linear_hessian',
    #               'learning_rate':                 learning_rate,
    #               'distiller':                     distiller,
    #               'use_closed_form_differentials': True,
    #               'risk_partitioning_objective':   True, # XXX
    #               }
    

    clfKwargs = { 'min_partition_size':            1,
                  'max_partition_size':            10, # 50
                  'row_sample_ratio':              0.5, # 0.50
                  'col_sample_ratio':              1.0,
                  'gamma':                         0.0, # 0.0025
                  'eta':                           0.0, # 0.05
                  'num_classifiers':               num_steps,
                  'use_constant_term':             False,
                  'solver_type':                   'linear_hessian',
                  'learning_rate':                 0.5, # 0.25
                  'distiller':                     distiller,
                  'use_closed_form_differentials': True,
                  'risk_partitioning_objective':   False,
                  }

    clf = OptimalSplitGradientBoostingClassifier( X_train,
                                                  y_train,
                                                  **clfKwargs
                                                 )

    clf.fit(num_steps)

    utils.oos_summary(clf, X_test, y_test)

