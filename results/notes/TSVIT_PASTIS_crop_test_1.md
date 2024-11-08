# Experiment Results Documentation

---

## 1. Dataset

- **Name**: PASTIS
- **Source**: Sentinel-2
- **Number of Samples**: 2468
- **Resolution**: 10m
- **Bands**: 10 sentinel bands
- **Area of Interest (AOI)**: To be added.

---

## 2. Preprocessing

- **Steps**: 
  - 
- **Tools Used**: Python script
- **Remarks**: Raw dataset

---

## 3. Satellite Information

- Satellite Name: Sentinel-2
- Date Range: September 2018 to December 2019
- Spatial Resolution: 10 meters
- Temporal Resolution: 10 - 11 days revisit

---

## 4. Model

- **Model Architecture**: TSVIT
- **Input Size**: 37 x 48 x 48
- **Number of Layers**: 
- **Number of Parameters**: NA
- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Loss Function**: Cross entropy loss
- **Batch Size**: 4
- **Epochs**: 50
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

- **Accuracy**:  75.85%
- **MIOU**: 45.24 %
- **F1 Score**: 58.37 %
- **Training Time**: 4 hours
- **Crop Results**: [Results](../csvs/pastis_1_results.csv)
- **Test Loss**:
- **Confusion Matrix**: ![Confusion Martix](../matrixes/confusion_matrix_PASTIS%201.png)


---

## 6. Observations

- **Strengths** Easy to run, and just cutted fields where the sample where less than 37.
- **Weaknesses**: I only cutted images, and in the cropping I used 48 x 48, so it doesn´t detects some classes as it should.
- **Further Improvements**: I think this model could work better if we had a more balanced dataset. Probably some techniques for this should work. Also maybe interpolate images 10 days as it says in the paper.

