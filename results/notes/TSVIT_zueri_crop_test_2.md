# Experiment Results Documentation

---

## 1. Dataset

- **Name**:  Zuericrop
- **Source**: Sentinel-2
- **Number of Samples**: 27977
- **Resolution**: 10m
- **Bands**: 9 sentinel bands
- **Area of Interest (AOI)**: To be added.

---

## 2. Preprocessing

- **Steps**: 
  - Class relabel to the 4th_tier_ENG, to have less classes and more generalization.
- **Tools Used**: Python script
- **Remarks**: Raw dataset

---

## 3. Satellite Information

- Satellite Name: Sentinel-2
- Date Range: January 2019 - December 2019
- Spatial Resolution: 10 meters
- Temporal Resolution: 3-5 day revisit

---

## 4. Model

- **Model Architecture**: TSVIT
- **Input Size**: 142 x 24 x 24
- **Number of Layers**: 
- **Number of Parameters**: NA
- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Loss Function**: Cross entropy loss
- **Batch Size**: 4
- **Epochs**: 10
- **Hyper Params**:{
    "patch_size": patch_size,
    "patch_size_time": 1,
    "patch_time": 4,
    "dim": 128,
    "temporal_depth": 6,
    "spatial_depth": 2,
    "channel_depth": 4,
    "heads": 4,
    "dim_head": 64,
    "dropout": 0.0,
    "emb_dropout": 0.0,
    "scale_dim": 4,
    "depth": 4,
}

---

## 5. Results

- **Accuracy**:  82.30%
- **Precision**: Nan
- **Recall**: Nan
- **F1 Score**: Nan
- **Training Time**: 7 hours
- **Crop Results**: [Results](../csvs/zueri_crop_2_results.csv)


---

## 6. Observations

- **Strengths** Easy to run and relabel. It looks like it has not bad accuracies for classes with lots of pixels.
- **Weaknesses**: It has really bad presicion with some crops and it seems overfitted, more if you consider the MIoU model. Probably because od the imbalance on the dataset
- **Further Improvements**: I think this model could work better if we had a more balanced dataset. Probably some techniques for this should work.


---
