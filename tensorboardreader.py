from tensorboard.backend.event_processing import event_accumulator
import xlsxwriter
import numpy as np
import os
import argparse

def Read_Tensorboard(path):
    ea = event_accumulator.EventAccumulator(path)
    ea.Reload()
    print(ea.scalars.Keys())
    val_loss = ea.scalars.Items("loss")
    val_F1 = ea.scalars.Items("F1")
    val_accuracy = ea.scalars.Items("accuracy")
    val_precision = ea.scalars.Items("precision")
    val_recall = ea.scalars.Items("recall")
    #val_acc2 = ea.scalars.Items("acc2")

    #print(val_dice)
    #print(len(val_dice))

    #print([(i.step,i.value) for i in val_dice])

    Epoch=[]
    Loss=[]
    F1=[]
    Accuracy=[]
    Precision=[]
    Recall=[]
    #Acc2=[]
    for i in range(len(val_loss)):
        #print(i+1,val_dice[i].value,val_IoU[i].value,val_mAcc[i].value,learning_rate[i].value)
        Epoch.append(i+1)
        Loss.append(val_loss[i].value)
        F1.append(val_F1[i].value)
        Accuracy.append(val_accuracy[i].value)
        Precision.append(val_precision[i].value)
        Recall.append(val_recall[i].value)
        #Acc2.append(val_acc2[i].value)
        
    return Epoch,Loss,F1,Accuracy,Precision,Recall

parser = argparse.ArgumentParser()
parser.add_argument("--root",type=str,default='./runs/')
parser.add_argument("--dir",type=str)

args = parser.parse_args()

root = args.root
directory = args.dir

#root = './runs/'
#directory = 'May09_00-40-39_container||BASELINE+CONTRA+ATTENTION||[G]=True[backbone]=ResNet50[dataset]=CIFAR100-7 - [batch_size]=128 - [dim_k]=2048 - [dim_v]=2048 - [n_heads]=16 - [lr]=0.001 - [alpha]=0.5'
filename = os.listdir(root+directory+'/')[0]
path = root + directory + '/' + filename

#path = './runs/May09_00-40-39_container||BASELINE+CONTRA+ATTENTION||[G]=True[backbone]=ResNet50[dataset]=CIFAR100-7 - [batch_size]=128 - [dim_k]=2048 - [dim_v]=2048 - [n_heads]=16 - [lr]=0.001 - [alpha]=0.5/events.out.tfevents.1652028039.container'

#e,l,f,a,p,r,a2=Read_Tensorboard(path)

#print(e)
#print(a)


def write_PR(Epoch,Loss,F1,Accuracy,Precision,Recall,output_filename):

    workbook = xlsxwriter.Workbook(output_filename, {'nan_inf_to_errors': True})
    #workbook = xlsxwriter.Workbook(output_filename)
    worksheet = workbook.add_worksheet()

    worksheet.activate()
    title = ['epoch','loss','F1','Accuracy','Precision','Recall']
    worksheet.write_row('A1',title)
    worksheet.write_row('A2',['best',min(Loss),max(F1),max(Accuracy),max(Precision),max(Recall)])

     # Start from the first cell below the headers.
    n_row = 3
    for i in range(len(Epoch)):
        insertData=[Epoch[i],Loss[i],F1[i],Accuracy[i],Precision[i],Recall[i]]
        row = 'A' + str(n_row)
        worksheet.write_row(row, insertData)
        n_row=n_row+1
    
    workbook.close()

e,l,f,a,p,r=Read_Tensorboard(path)
write_PR(e,l,f,a,p,r,'./xlsxoutput/'+directory+'.xlsx')