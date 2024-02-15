# **Brain Tumour Image Segmentation**

In this study, a U-net model and a morphological image processing technique were used to improve the segmentation of brain MR images. The U-net model was chosen because it can accurately segment images by capturing both local and global picture properties. The study aimed to improve the segmentation results by modifying the borders of the segmented regions by incorporating a morphological image processing method into the pipeline. The administration of big and numerous image databases was the project's main obstacle. A downsampling technique was used to solve this problem, enabling effective data processing while maintaining crucial data required for precise segmentation. The study's findings showed a notable improvement in the precision of tumour areas' detection in brain MR images.

---

## **DATASET**

The LGG (Brain Tumor) Segmentation Dataset is a collection of brain magnetic resonance (MR) images and corresponding manual segmentation masks for FLAIR abnormality. The dataset was obtained from the Cancer Imaging Archive (TCIA) and consists of images from 110 patients included in The Cancer Genome Atlas (TCGA) lower-grade glioma collection. Which we found through Kaggle. 

The dataset is provided in the form of TIFF images. Each image is accompanied by a corresponding segmentation mask that highlights the abnormal regions in the FLAIR images. In total, the dataset comprises 7,858 images, with 3,929 images representing the original MRI scans and the remaining 3,929 images serving as the corresponding masks for the original images. The dataset is organized into 110 folders, where each folder is named after the case ID and contains information about the source institution. This organization facilitates easy access to images and their corresponding metadata, allowing researchers and practitioners to analyze the data efficiently.

---

## **RESULTS**
The U-Net model alone demonstrates its capability to generate accurate predictions for object segmentation. The U-Net predictions capture the general shape and location of the objects of
interest, aligning reasonably well with the boundaries outlined in the ground truth masks. This highlights the effectiveness of U-Net in learning intricate patterns and features from the training data. In some cases, the U-Net predictions may exhibit gaps or holes within the segmented objects, failing to completely fill them in. To address these shortcomings, the hybrid approach incorporates morphological image processing techniques after obtaining the U-Net predictions. By applying dilation and erosion operations, the hybrid approach aims to refine and improve the predicted masks. The morphological operations help fill in the gaps, connect fragmented regions, and smoothen the boundaries of the objects.

In summary, while the hybrid approach of combining U-Net with morphological image processing may not result in significant improvements in every case, it does offer benefits such as filling gaps
and smoothening boundaries. The choice of whether to incorporate such post-processing techniques depends on the specific requirements of the task at hand and the characteristics of the dataset.
Experimenting with different combinations of models and image processing methods can lead tofurther advancements in object segmentation and provide valuable insights for future research in this field.

---

## **CONTENT**

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
