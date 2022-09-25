import os


ROOT_PATH_ = "/pool0/home/karthik"
ROOT_PATH = os.path.join(ROOT_PATH_, "files_from_lab_imac/OMOP_EHR")
RESULT_DIR = os.path.join(ROOT_PATH, "temporal_spokesig", "data")

TRAIN_DATA_PATH = os.path.join("data" , "train")
TEST_DATA_PATH = os.path.join("data" , "test")
MODEL_PATH = os.path.join("data", "pretrained_models")

TRAIN_DATA_SCORE_PATH = os.path.join("data", "tandem_train_data_prediction_score.csv")

TEMPORAL_MODEL_PATH = os.path.join(MODEL_PATH, "temporal_model.joblib")
TEMPORAL_TRAIN_DATA_PATH = os.path.join(TRAIN_DATA_PATH, "train_data_temporal.npy")
TEMPORAL_TEST_DATA_PATH = os.path.join(TEST_DATA_PATH, "temporal_test_data.npy")

NON_TEMPORAL_MODEL_PATH = os.path.join(MODEL_PATH, "non_temporal_model.joblib")
NON_TEMPORAL_TRAIN_DATA_PATH = os.path.join(TRAIN_DATA_PATH, "train_data_non_temporal.npy")
NON_TEMPORAL_TEST_DATA_PATH = os.path.join(TEST_DATA_PATH, "non_temporal_test_data.npy")

TANDEM_MODEL_PATH = os.path.join(MODEL_PATH, "logistic_classifier.h5")

TRAIN_METADATA_PATH = os.path.join(TRAIN_DATA_PATH, "train_metadata.csv")
TEST_METADATA_PATH = os.path.join(TEST_DATA_PATH, "test_metadata.csv")
