# Experiment Results Documentation

---

## 1. Dataset

- **Name**:  Zuericrop
- **Source**: Sentinel-2
- **Number of Samples**: 12000
- **Resolution**: 10m
- **Bands**: 9 sentinel bands
- **Area of Interest (AOI)**: To be added.

---

## 2. Preprocessing

- **Steps**: 
  - Relabel all classes to 0 (background class) except the classes that are more balanced: 60,10,58,15
  - Normalize band values by multiplying by 0.0001.
  - Remove all the images that have a less than 100 non zero pixels.
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

- **Accuracy**:   83.56 %
- **MIOU**: 65.53 %
- **F1 Score**: 78.56 %
- **Training Time**: 13 hours
- **Crop Results**: [Results](../csvs/zueri_crop_5_results.csv)
- **Test Loss**: 0.6512
- **Confusion Matrix**: ![Confusion Martix](../images/confusion_matrix_zuericrop%205.png)


---

## 6. Observations

- **Strengths** Easy to run and relabel.The dataset looks unbalanced as hell this is probably because of the background class.
- **Weaknesses**: Probably some of the classes have other labels with the same crop as 0 that's why sometimes it says labels a lot as 0.
- **Further Improvements**: Add more classes and merge some classes that are probably off.

