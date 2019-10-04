import os

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

TRAINED_PATH = os.path.join(BASE_PATH, "output/trained_model")

EXPS_PATH = os.path.join(BASE_PATH, "output/experiments")

DATA_DIR = os.path.join(BASE_PATH, 'datasets')

EMB_DIR = os.path.join(BASE_PATH, 'embeddings')


PAD = 0

