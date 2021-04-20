
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

Our proposed approach is to try a multitude of single classifiers with different loss functions for this problem, see the best performance we can get there, and then utilize the best performing loss function from this test to try and get even better ensemble performance

**How to run our codes**
These codes have 3 ways they can be run, and a number of command line parameters to run the main.py script. These are run in google colab, so you would need to zip your file directory from this git repo into a .zip or .tar file, put it in your google drive, and then 

![Tutorial 1](/code/colab_1.PNG)
![Tutorial 2](/code/colab_2.PNG)

Single classifier example: 

!python main.py --datapath /content/chest_xray --loss SmoothCrossEntropy --num_labels 2 --batch_size 64 --steps 500 --save_interval 100 --net=resnet50 --runs=1

Ensemble majority voting example: 
!python main.py --datapath /content/chest_xray --loss SmoothCrossEntropy --num_labels 2 --runs 1 --batch_size 64 --steps 1000 --save_interval 200 --ensemble googlenet resnet101 vgg alexnet inception_v3 --runs 1

Ensemble adaboost example: 
!python main.py --datapath /content/chest_xray --loss SmoothCrossEntropy --num_labels 2 --runs 1 --batch_size 64 --steps 1000 --save_interval 200 --ensemble adaBoost googlenet resnet101 vgg alexnet inception_v3 --runs 1

Command line parameters: 
'--steps', type=int, default=3000, Number of iterations to train for

'--loss', type=str, default='CrossEntropy', choices=['CrossEntropy', 'FocalLoss','SmoothCrossEntropy', 'ASoftmaxLoss'], 

'--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.001)'

'--multi', type=float, default=0.1, metavar='MLT', help='learning rate multiplication'

'--T', type=float, default=0.05, metavar='T', help='temperature (default: 0.05)'

'--lamda', type=float, default=0.1, metavar='LAM', help='value of lamda'

'--save_check', action='store_true', default=True, help='save checkpoint or not'

'--datapath', type=str, default='./data', help='dir to save training data'

'--checkpath', type=str, default='./save_model', help='dir to save checkpoint'

'--seed', type=int, default=1, metavar='S', help='random seed (default: 1)'

'--log_interval', type=int, default=100, metavar='N', help='how many batches to wait before logging training status'

'--save_interval', type=int, default=500, metavar='N', help='how many batches to wait before saving a model'

'--num_labels', type=int, default=2, help='number of labels (default: 2)'

'--net', type=str, default='resnet50', help='which network to use'

'--patience', type=int, default=100, metavar='S', help='early stopping to wait for improvment '
                            'before terminating. (default: 5 (5000 iterations))'
                            
'--early', action='store_false', default=True, help='early stopping on validation or not'

'--batch_size', type=int, default=16, metavar='S', help='random seed (default: 1)'

'--worker', type=int, default=3, metavar='S', help='random seed (default: 1)'

'--runs', type=int, default=1, help='run the whole classifier (train/test) this many times and average the test accuracy'

'--ensemble', '--list', nargs='+', help= 'List of ensemble classifiers to use',required=False
    

**Evaluation Results**

Single model classifiers: 

![Table 1](/code/table.PNG)

From this we found smooth crossentropy to be the best loss function overall, and used this for some ensemble method tests:

Majority voting (2 class problem), smooth crossentropy, vgg resnet50 alexnet resnext inception_v3, 97.02% test accuracy

Adaboost (2 class): smooth crossentropy, vgg resnet50 alexnet resnext inception_v3, 97.3333333% test accuracy

Adaboost (2 class): smooth crossentropy ,vgg resnet50 alexnet resnext inception_v3 googlenet resnet101, 97.646% test accuracy



