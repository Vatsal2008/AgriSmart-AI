# AgriSmart training dataset

Built 2026-09-10 14:35 UTC by the `agrismart-data-prep` notebook: 28 classes, 45583 images.

| Split | Images | What it is |
|---|---|---|
| train | 35257 | PlantVillage lab photos (split by leaf) + field photos from PlantWild v1/v2 and PlantSeg |
| val_lab | 7520 | PlantVillage photos of leaves that are not in train |
| val_field | 2806 | PlantDoc field photos, never trained on |

Training field photos that look like any PlantDoc photo were removed (DINOv2 cosine >= 0.93 or pHash distance <= 10).

## Sources and licences
- **PlantVillage**: Mohanty, Hughes & Salathé (2016), *Using deep learning for image-based plant disease detection*, Frontiers in Plant Science. Kaggle `abdallahalidev/plantvillage-dataset` (CC BY-NC-SA 4.0); leaf map and split from Hugging Face `mohanty/PlantVillage` (CC BY-SA 3.0).
- **PlantDoc**: Singh et al. (2020), *PlantDoc: A Dataset for Visual Plant Disease Detection*, CoDS-COMAD, doi:10.1145/3371158.3371196. Kaggle `nirmalsankalana/plantdoc-dataset` (CC BY 4.0). Validation only.
- **PlantWild v1/v2**: Wei et al. (2024), *Benchmarking In-the-wild Multimodal Plant Disease Recognition and A Versatile Baseline*, ACM Multimedia, arXiv:2408.03120. Hugging Face `uqtwei2/PlantWild` (CC BY-NC-ND 4.0: non-commercial, no redistribution).
- **PlantSeg**: Wei et al. (2025), *A Large-Scale In-the-wild Dataset for Plant Disease Segmentation*, Scientific Data. Kaggle `weitianqi/plantseg`, Zenodo 14935094 (CC BY 4.0).

Keep this output private: it contains PlantWild images, whose licence forbids redistribution.
