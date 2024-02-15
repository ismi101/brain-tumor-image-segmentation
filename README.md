# Brain Tumour Image Segmentation

**ABSTRACT**

In this study, a U-net model and a morphological image processing technique were used to improve the segmentation of brain MR images. The U-net model was chosen because it can accurately segment images by capturing both local and global picture properties. The study aimed to improve the segmentation results by modifying the borders of the segmented regions by incorporating a morphological image processing method into the pipeline. The administration of big and numerous image databases was the project's main obstacle. A downsampling technique was used to solve this problem, enabling effective data processing while maintaining crucial data required for precise segmentation. The study's findings showed a notable improvement in the precision of tumour areas' detection in brain MR images. The dice similarity
coefficient (DSC), one of the evaluation criteria, supported the success of the combined strategy. The accurate localisation of tumour locations made possible by the U-net model and morphological image processing method has the potential to aid medical practitioners in improving tumour diagnosis and treatment planning

---

**Dataset Description**

The LGG (Brain Tumor) Segmentation Dataset is a collection of brain magnetic resonance (MR) images and corresponding manual segmentation masks for FLAIR abnormality. The dataset was obtained from the Cancer Imaging Archive (TCIA) and consists of images from 110 patients included in The Cancer Genome Atlas (TCGA) lower-grade glioma collection. Which we found through Kaggle. The dataset is provided in the form of TIFF images. Each image is accompanied by
a corresponding segmentation mask that highlights the abnormal regions in the FLAIR images. In total, the dataset comprises 7,858 images, with 3,929 images representing the original MRI scans and the remaining 3,929 images serving as the corresponding masks for the original images. The dataset is organized into 110 folders, where each folder is named after the case ID and contains information about the source institution. This organization facilitates easy access to images and their corresponding metadata, allowing researchers and practitioners to analyze the data efficiently.



---

**CONTENT**

**1. Importing Libraries**

**2. Loading Dataset**

**3. EDA & Visualization**

**4. Dataset Partitioning**

**5. Data Pre-processing**
- Data Augmentation
- Normalization and Mask Thresholding
- Data Generators

**6. Evaluation Metrics**

**7. Model Development**
- Model Architecture (U-Net)
- Model Compilation
- Model Training

**8. Model Evaluation**

**9. Morphological Image Processing**
- Thresholding and Morphological Operations
- Applying Morphological Operations to Predicted Masks

**10. Prediction Results (U-Net Only vs Hybrid)**
