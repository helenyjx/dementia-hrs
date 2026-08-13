import os


DIR_RESOURCES = 'data'
DIR_RO = os.path.join(DIR_RESOURCES, 'read_only')
DIR_RW = os.path.join(DIR_RESOURCES, 'read_write')
#FILENAME = os.path.join(DIR_RO, 'randhrs1992_2020v1.dta')
FILENAME = os.path.join(DIR_RO, 'randhrs1992_2022v1.dta')

SELF_DEM_COLS = [
    'imrc',  # immediate recall of a list of 10 words
    'dlrc',  # delayed recall of a list of 10 words
    'ser7',  # five trials of serial 7s
    'bwc20',  # backward counting
]
PROXY_DEM_COLS = [
    'prmem',  # proxy rating of respondent memory
    # 'moneya', 'medsa', 'mealsa', 'phonea', 'shopa', # 5 IADLs (any difficulties)
    'iadl5a',
    'prfin',  # predicted finish: interviewer’s assessment of the respondent’s difficulty completing
]

SEED = 42

RESULTS_DIR = 'results'
#CHECKPOINTS_DIR = 'checkpoints'
#IMPUTERS_DIR = os.path.join(CHECKPOINTS_DIR, 'imputers')
#FSELECTORS_DIR = os.path.join(CHECKPOINTS_DIR, 'fselectors')
#PREDICTORS_DIR = os.path.join(CHECKPOINTS_DIR, 'predictors')
