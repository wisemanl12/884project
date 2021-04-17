




**Introduction**

The idea behind our project is to try and outperform the classification accuracy on the   achieved in the following two papers: 

[Transfer learning with chest X-rays for ER patient classification](https://www.mdpi.com/2076-3417/10/2/559) (1)
Which was published in 2020, and
[Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning](https://pubmed.ncbi.nlm.nih.gov/29474911/) (2)
which was published in feb 2018
In these papers, they got classification accuracy of 

Reference | Model | Accuracy
------------ | ------------ | -------------
1 | Ensemble of multiple models | 96.4%
1 | Resnet18 | 94.3%
2 | Inception V3 | 92.8%


For this, we used google colab and decided to still use transfer learning on this projects dataset, which can be found at 
(https://www.rsna.org/education/ai-resources-and-training/ai-image-challenge/RSNA-Pneumonia-Detection-Challenge-2018)

**Proposed approach**
Source (1) got their highest accuracy using a majority voting ensemble of the pretrained AlexNet, Densenet121, Inception_V3, ResNext, GoogleNet

**How to run our codes**

**Evaluation Results**



