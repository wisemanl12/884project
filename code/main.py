from __future__ import print_function
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from model.resnet import resnet34
from model.basenet import AlexNetBase, VGGBase, ResBase50, ResBase101, Predictor
from utils.utils import weights_init
from utils.lr_schedule import inv_lr_scheduler
from utils.return_dataset import return_dataset
from utils.loss import entropy, crossentropy, smooth_crossentropy, focal_loss, asoftmax_loss
import torchvision.models as models



#train function, along with an optional weights parameter for sample weighting
#in the case that we want to weight different samples differently
def train(args,weights=None):
    if os.path.exists(args.checkpath) == False:
        os.mkdir(args.checkpath)

    # 1. get dataset
    train_loader, val_loader, test_loader, class_list = return_dataset(args)

    # 2. generator
    if args.net == 'resnet50':
        G = ResBase50()
        inc = 2048
    elif args.net == 'resnet101':
        G = ResBase101()
        inc = 2048        
    elif args.net == "alexnet":
        G = AlexNetBase()
        inc = 4096
    elif args.net == "vgg":
        G = VGGBase()
        inc = 4096
    elif args.net == "inception_v3":
        G = models.inception_v3(pretrained=True) 
        inc = 1000
    elif args.net == "googlenet":
        G = models.googlenet(pretrained = True)
        inc = 1000
    elif args.net == "densenet":
        G = models.densenet161(pretrained = True)
        inc = 1000
    elif args.net == "resnext":
        G = models.resnext50_32x4d(pretrained = True)
        inc = 1000
    elif args.net == "squeezenet":
        G = models.squeezenet1_0(pretrained = True)
        inc = 1000
    else:
        raise ValueError('Model cannot be recognized.')

    params = []
    for key, value in dict(G.named_parameters()).items():
        if value.requires_grad:
            if 'classifier' not in key:
                params += [{'params': [value], 'lr': args.multi,
                            'weight_decay': 0.0005}]
            else:
                params += [{'params': [value], 'lr': args.multi * 10,
                            'weight_decay': 0.0005}]
    G.cuda()
    G.train()

    # 3. classifier
    F = Predictor(num_class=len(class_list), inc=inc, temp=args.T)
    weights_init(F)
    F.cuda()
    F.train()  

    # 4. optimizer
    optimizer_g = optim.SGD(params, momentum=0.9,
                            weight_decay=0.0005, nesterov=True)
    optimizer_f = optim.SGD(list(F.parameters()), lr=1.0, momentum=0.9,
                            weight_decay=0.0005, nesterov=True) 
    optimizer_g.zero_grad()
    optimizer_f.zero_grad()

    param_lr_g = []
    for param_group in optimizer_g.param_groups:
        param_lr_g.append(param_group["lr"])
    param_lr_f = []
    for param_group in optimizer_f.param_groups:
        param_lr_f.append(param_group["lr"])

    # 5. training
    data_iter_train = iter(train_loader)
    len_train = len(train_loader)
    best_acc = 0
    for step in range(args.steps):
        # update optimizer and lr
        optimizer_g = inv_lr_scheduler(param_lr_g, optimizer_g, step,
                                       init_lr=args.lr)
        optimizer_f = inv_lr_scheduler(param_lr_f, optimizer_f, step,
                                       init_lr=args.lr)
        lr = optimizer_f.param_groups[0]['lr']
        if step % len_train == 0:
            data_iter_train = iter(train_loader)

        # forwarding
        data = next(data_iter_train)        
        im_data = data[0].cuda()
        gt_label = data[1].cuda()

        feature = G(im_data)
        if args.net == 'inception_v3': #its not a tensor output but some 'inceptionOutput' object
          feature = feature.logits #get the tensor object
        if args.loss=='CrossEntropy': 
            #call with weights if present 
            loss = crossentropy(F, feature, gt_label, None if (weights == None) else weights[step % len_train])
            #although the weights might be defaulting to none
        elif args.loss=='FocalLoss':
            loss = focal_loss(F, feature, gt_label, None if (weights == None) else weights[step % len_train])
        elif args.loss=='ASoftmaxLoss':
            loss = asoftmax_loss(F, feature, gt_label, None if (weights == None) else weights[step % len_train])
        elif args.loss=='SmoothCrossEntropy':
            loss = smooth_crossentropy(F, feature, gt_label, None if (weights == None) else weights[step % len_train])
        else:
            print('To add new loss')         
        loss.backward()

        # backpropagation
        optimizer_g.step()
        optimizer_f.step()       
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()
        G.zero_grad()
        F.zero_grad()

        if step%args.log_interval==0:
            log_train = 'Train iter: {} lr{} Loss Classification: {:.6f}\n'.format(step, lr, loss.data)
            print(log_train)
        if step and step%args.save_interval==0:
            # evaluate and save
            acc_val = eval(val_loader, G, F, class_list)     
            G.train()  
            F.train()  
            if args.save_check and acc_val >= best_acc:
                best_acc = acc_val
                print('saving model')
                print('best_acc: '+str(best_acc) + '  acc_val: '+str(acc_val))
                torch.save(G.state_dict(), os.path.join(args.checkpath,
                                        "G_net_{}_loss_{}.pth".format(args.net, args.loss)))
                torch.save(F.state_dict(), os.path.join(args.checkpath,
                                        "F_net_{}_loss_{}.pth".format(args.net, args.loss)))
    if (weights is not None):
      print("computing error rate")
      error_rate = eval_adaboost_error_rate(train_loader, G, F, class_list, weights)
      model_importance = torch.log((1-error_rate)/error_rate)/2
      #now update the weights
      print("updating weights")
      update_weights_adaboost(train_loader, G, F, class_list, weights, model_importance)
      return error_rate, model_importance

#function to update the weights for the adaboost classifier after once classifier has been trained
#and its importance alpha has been determined
def update_weights_adaboost(loader, G, F, class_list, weights, alpha):
    G.eval()
    F.eval()
    num_class = len(class_list)
    confusion_matrix = np.zeros((num_class, num_class))
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            w = weights[batch_idx]
            im_data = data[0].cuda()
            gt_labels = data[1].cuda()
            feat = G(im_data)
            output = F(feat)            
            pred = output.data.max(1)[1]
            weights[batch_idx] = weights[batch_idx]*adaboost_update_formula(alpha,pred,gt_labels.data)

def adaboost_update_formula(alpha, pred, labels):
    #using update formula that either lowers weight if it was predicted right, or increases if 
    #predicted wrong
    result = torch.zeros(pred.shape).cuda()
    count = 0
    for y,fx in zip(pred,labels):
      result[count] = torch.exp(-alpha) if (y==fx) else torch.exp(alpha)
      count+=1
    return result
#function to evaluate the error rate of thsi classifier based on the definition
#for adaboost
def eval_adaboost_error_rate(loader, G, F, class_list, weights):
    G.eval()
    F.eval()
    error_rate = 0
    size = 0
    num_class = len(class_list)
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            w = weights[batch_idx]
            im_data = data[0].cuda()
            gt_labels = data[1].cuda()
            feat = G(im_data)
            output = F(feat)            
            pred = output.data.max(1)[1]
            size += im_data.size(0)      
            error_rate += ((pred.ne(gt_labels.data))*w[0:pred.shape[0]]).cpu().sum()
            
            
    return error_rate/torch.sum(weights)

def eval(loader, G, F, class_list):
    G.eval()
    F.eval()
    test_loss = 0
    correct = 0
    size = 0
    num_class = len(class_list)
    criterion = nn.CrossEntropyLoss().cuda()
    confusion_matrix = np.zeros((num_class, num_class))
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            im_data = data[0].cuda()
            gt_labels = data[1].cuda()
            if len(class_list) == 3:
                gt_labels[gt_labels>1]=2
            feat = G(im_data)
            output = F(feat)            
            pred = output.data.max(1)[1]
            for t, p in zip(gt_labels.view(-1), pred.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            size += im_data.size(0)   
            if len(class_list) == 3:
                pred[pred>1]=2                 
            correct += pred.eq(gt_labels.data).cpu().sum()
            test_loss += criterion(output, gt_labels) / len(loader)
            
    return 100. * float(correct) / size



def test(args):
    # 1. get dataset
    train_loader, val_loader, test_loader, class_list = return_dataset(args)

    # 2. generator
    if args.net == 'resnet50':
        G = ResBase50()
        inc = 2048
    elif args.net == 'resnet101':
        G = ResBase101()
        inc = 2048        
    elif args.net == "alexnet":
        G = AlexNetBase()
        inc = 4096
    elif args.net == "vgg":
        G = VGGBase()
        inc = 4096
    elif args.net == "inception_v3":
        G = models.inception_v3(pretrained=True) 
        inc = 1000
    elif args.net == "googlenet":
        G = models.googlenet(pretrained = True)
        inc = 1000
    elif args.net == "densenet":
        G = models.densenet161(pretrained = True)
        inc = 1000
    elif args.net == "resnext":
        G = models.resnext50_32x4d(pretrained = True)
        inc = 1000
    elif args.net == "squeezenet":
        G = models.squeezenet1_0(pretrained = True)
        inc = 1000
    else:
        raise ValueError('Model cannot be recognized.')
    G.cuda()
    G.train()

    # 3. classifier
    F = Predictor(num_class=len(class_list), inc=inc, temp=args.T)
    F.cuda()
    F.train()  

    # 4. load pre-trained model
    G.load_state_dict(torch.load(os.path.join(args.checkpath,
                                        "G_net_{}_loss_{}.pth".format(args.net, args.loss))))
    F.load_state_dict(torch.load(os.path.join(args.checkpath,
                                        "F_net_{}_loss_{}.pth".format(args.net, args.loss))))
    # 5. testing
    acc_test = eval(test_loader, G, F, class_list)
    print('Testing accuracy: {:.3f}\n'.format(acc_test))    
    return acc_test 


#function to take in a batch of image data, a list of classifiers,
#and generate a final prediction using either majority voting or ada boosting
#(with precomputed alpha weights)
#if alphas inputted to this function are none, then it assumes we are doing majority voting
def ensemble_result(args,G_list,F_list, im_data,alphas):
    predictions = torch.tensor(np.zeros((len(G_list),args.batch_size))).cuda()
    count = 0
    #based on whether or not alphas is None or an array, pick how we use the
    #classifier outputs
    pred = []
    if (alphas is None):
      for G,F in zip(G_list,F_list):
        feat = G(im_data)
        output = F(feat)
        predictions[count,:]=output.data.max(1)[1]
        count+=1
        #majority voting
      pred = torch.mode(predictions,0) 
      pred = pred.values
    else:
      print("Alphas: " , alphas, " adaboost eval")
      output_overall = []
      for G,F in zip(G_list,F_list):
        feat = G(im_data)
        output = F(feat)
        if (count == 0):
          output_overall = alphas[count]*output.data
          count+=1
        else:
          output_overall+= alphas[count]*output.data
          count+=1
      pred = output_overall.max(1)[1]
    return pred
#ensemble variant of evaluating the model using a list of predictors and classifiers
#calls a function based on which ensemble method is being used
def eval_ensemble(args,loader, G_list, F_list, class_list,alphas=None):
    for G,F in zip(G_list,F_list):
      G.eval()
      F.eval()
    
    #test_loss = 0
    correct = 0
    size = 0
    num_class = len(class_list)
    criterion = nn.CrossEntropyLoss().cuda()
    confusion_matrix = np.zeros((num_class, num_class))
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            im_data = data[0].cuda()
            gt_labels = data[1].cuda()
            if len(class_list) == 3:
                gt_labels[gt_labels>1]=2      
            #now based on the ensemble method used, we utilize this data

            pred = ensemble_result(args,G_list,F_list, im_data, alphas)
            for t, p in zip(gt_labels.view(-1), pred.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            size += im_data.size(0)   
            if len(class_list) == 3:
                pred[pred>1]=2                 
            correct += pred.eq(gt_labels.data).cpu().sum()
            #test_loss += criterion(output, gt_labels) / len(loader)
            
    return 100. * float(correct) / size

def test_ensemble(args, alphas=None):
  # 1. get dataset
  train_loader, val_loader, test_loader, class_list = return_dataset(args)
  print("Loading in ")
  # 2. generator
  G_list = [] #use a list of models
  F_list = [] #use a list of predictors, one for each classifier in args.ensemble

  for classifier in args.ensemble:
    print("classifier: ", classifier)
    if classifier == 'resnet50':
        G = ResBase50()
        inc = 2048
    elif classifier == 'resnet101':
        G = ResBase101()
        inc = 2048        
    elif classifier == "alexnet":
        G = AlexNetBase()
        inc = 4096
    elif classifier == "vgg":
        G = VGGBase()
        inc = 4096
    elif classifier == "inception_v3":
        G = models.inception_v3(pretrained=True) 
        inc = 1000
    elif args.net == "googlenet":
        G = models.googlenet(pretrained = True)
        inc = 1000
    elif args.net == "densenet":
        G = models.densenet161(pretrained = True)
        inc = 1000
    elif args.net == "resnext":
        G = models.resnext50_32x4d(pretrained = True)
        inc = 1000
    elif args.net == "squeezenet":
        G = models.squeezenet1_0(pretrained = True)
        inc = 1000
    else:
        raise ValueError('Model cannot be recognized.')
    G.cuda()
    G.train()

    # 3. classifier
    F = Predictor(num_class=len(class_list), inc=inc, temp=args.T)
    F.cuda()
    F.train()  

    # 4. load pre-trained model
    G.load_state_dict(torch.load(os.path.join(args.checkpath,
                                        "G_net_{}_loss_{}.pth".format(classifier, args.loss))))
    F.load_state_dict(torch.load(os.path.join(args.checkpath,
                                        "F_net_{}_loss_{}.pth".format(classifier, args.loss))))
    G_list.append(G)
    F_list.append(F)
  # 5. testing
  acc_test = eval_ensemble(args,test_loader, G_list, F_list, class_list,alphas)
  print('Testing accuracy: {:.3f}\n'.format(acc_test))      
  return acc_test                     



if __name__ == "__main__":   
    # Training settings
    parser = argparse.ArgumentParser(description='X-Ray Chest Classification')
    parser.add_argument('--steps', type=int, default=3000, metavar='N', help='maximum number of iterations to train (default: 50000)')
    parser.add_argument('--loss', type=str, default='CrossEntropy', choices=['CrossEntropy', 'FocalLoss','SmoothCrossEntropy', 'ASoftmaxLoss'], help='loss functions to be tested')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.001)')
    parser.add_argument('--multi', type=float, default=0.1, metavar='MLT', help='learning rate multiplication')
    parser.add_argument('--T', type=float, default=0.05, metavar='T', help='temperature (default: 0.05)')
    parser.add_argument('--lamda', type=float, default=0.1, metavar='LAM', help='value of lamda')
    parser.add_argument('--save_check', action='store_true', default=True, help='save checkpoint or not')
    parser.add_argument('--datapath', type=str, default='./data', help='dir to save training data')
    parser.add_argument('--checkpath', type=str, default='./save_model', help='dir to save checkpoint')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=100, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--save_interval', type=int, default=500, metavar='N', help='how many batches to wait before saving a model')
    parser.add_argument('--num_labels', type=int, default=2, help='number of labels (default: 2)')

    parser.add_argument('--net', type=str, default='resnet50', help='which network to use')
    parser.add_argument('--patience', type=int, default=100, metavar='S', help='early stopping to wait for improvment '
                            'before terminating. (default: 5 (5000 iterations))')
    parser.add_argument('--early', action='store_false', default=True, help='early stopping on validation or not')
    parser.add_argument('--batch_size', type=int, default=16, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--worker', type=int, default=3, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--runs', type=int, default=1, help='run the whole classifier (train/test) this many times and average the test accuracy')
    #add argument for ensemble methods
    parser.add_argument('--ensemble', '--list', nargs='+', help= 'List of ensemble classifiers to use',required=False)
    args = parser.parse_args()


    total_acc = 0
    for i in range(0,args.runs):
      if ( (args.ensemble is not None) and len(args.ensemble) > 1):
        #average multiple models, check if we are doing adaboost or regular
        print("Ensemble method being used")
        if (args.ensemble[0] == 'adaBoost'):
          print("doing ada boost version")
          # 1. get size of dataset for the weights
          train_loader, val_loader, test_loader, class_list = return_dataset(args)
          len_train = len(train_loader)
          print("size of train: ", len_train)
          #create a tensor of ones for initial weights
          weights = torch.ones(len_train,args.batch_size).cuda()
          #since were doing batch size, each call to the loss function is with
          #one batch. So using a matrix of weights instead of an array might be more clear
          alphas = [] #importance of each classifier

          for classifier in args.ensemble[1::]:
            print(classifier)
            args.net = classifier
            error_rate,local_alpha = train(args,weights) 
            print("error rate: ", error_rate)
            print("alpha m: ", local_alpha)
            
            alphas.append(local_alpha)
          #erase the adaboost element in the args for this part, leave alphas
          args.ensemble = args.ensemble[1::]
          total_acc = total_acc + test_ensemble(args,alphas)
        else:
          for classifier in args.ensemble:
            print(classifier)
            #train classifier
            args.net = classifier
            train(args)
          total_acc = total_acc + test_ensemble(args)
      else:
        train(args)                              
        total_acc+=test(args)
    print("pre devision total acc: ")
    total_acc = total_acc / args.runs   #divide the sum of the accuracy by number of runs

    #print out final average accuracy:
    print("Final average accuracy after {} runs: {} ".format(args.runs, total_acc)) 
