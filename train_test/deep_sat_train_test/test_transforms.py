from transforms import PASTIS_segmentation_transform
import pickle

model_config = {
    "img_res": 24,
    "labels": 19,
    "max_seq_len": 60,
}


transform_pipeline = PASTIS_segmentation_transform(model_config, True)

# Get pickle file from the dataset
sample = pickle.load(open("pickles/10110_15.pickle", "rb"))

transformed_sample = transform_pipeline(sample)


print(transformed_sample["inputs"].shape)
