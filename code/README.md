
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
Source (1) got their highest accuracy using a majority voting ensemble of the pretrained AlexNet, Densenet121, Inception_V3, ResNext, and GoogleNet, with an accuracy on the test set of 96.4%

**How to run our codes**
These codes have 3 ways they can be run, and a number of command line parameters to run the main.py script. These are run in google colab, so you would need to zip your file directory from this git repo into a .zip or .tar file, put it in your google drive, and then 

[TODO: INSERT ALL THE COLAB STEPS (SCREENSHOT)]
![Tutorial 1](/code/colab_1.PNG)

Single classifier example: 

!python main.py --datapath /content/chest_xray --loss SmoothCrossEntropy --num_labels 2 --batch_size 64 --steps 500 --save_interval 100 --net=resnet50 --runs=1

Ensemble majority voting example: 
!python main.py --datapath /content/chest_xray --loss SmoothCrossEntropy --num_labels 2 --runs 1 --batch_size 64 --steps 1000 --save_interval 200 --ensemble googlenet resnet101 vgg alexnet inception_v3 --runs 1

Ensemble adaboost example: 
!python main.py --datapath /content/chest_xray --loss SmoothCrossEntropy --num_labels 2 --runs 1 --batch_size 64 --steps 1000 --save_interval 200 --ensemble adaBoost googlenet resnet101 vgg alexnet inception_v3 --runs 1

Command line parameters: 
**Evaluation Results**
