import os
import shutil

from tqdm import tqdm

cur_path = os.getcwd()
orig_path = os.path.join(cur_path, 'flower_dataset')
targ_path = os.path.join(cur_path, 'dataset')
train_path = os.path.join(targ_path, 'train')
val_path = os.path.join(targ_path, 'val')

os.mkdir(train_path)
os.mkdir(val_path)

Classes_file = open(targ_path+'\\classes.txt', 'w')
Train_file = open(targ_path+'\\train.txt', 'w')
Val_file = open(targ_path+'\\val.txt', 'w')

Classes = os.listdir(orig_path)
for i, clas in tqdm(enumerate(Classes)):
    orig_class_path = os.path.join(orig_path, clas)
    targ_train_class_path = os.path.join(train_path, clas)
    targ_val_class_path = os.path.join(val_path, clas)
    os.mkdir(targ_train_class_path)
    os.mkdir(targ_val_class_path)
    Classes_file.write(clas+'\n')
    Samples = os.listdir(orig_class_path)
    for j, sample in enumerate(Samples):
        orig_data_path = os.path.join(orig_class_path, sample)
        if j%5 == 0:
            targ_data_path = os.path.join(targ_val_class_path, sample)
            Val_file.write(clas+'/'+sample+' '+str(i)+'\n')
        else:
            targ_data_path = os.path.join(targ_train_class_path, sample)
            Train_file.write(clas+'/'+sample+' '+str(i)+'\n')
        shutil.copy(orig_data_path, targ_data_path)

Classes_file.close()
Train_file.close()
Val_file.close()
