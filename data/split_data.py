import glob
import os
import shutil

testing_data = glob.glob('oasis/testing/*')
training_data = glob.glob('oasis/training/*')
validation_data = glob.glob('oasis/validation/*')
all_data = glob.glob('oasis/*')

validation = ['0292','0262','0261','0277','0295','0272','0263','0293','0256','0264','0268',
 '0296','0266','0278','0305','0288','0280','0299','0286','0281','0279','0265','0287','0283','0271','0274',
'0258','0259','0289','0282','0275','0298','0294','0260','0303','0290','0273','0285','0302','0270','0307', '0269','0301','0291','0284','0267','0300','0304']
testing = ['0392','0396','0420','0402','0386','0382','0419','0430','0383','0424','0398','0384','0415','0400','0409','0411','0390','0381','0428','0403','0405','0385','0389',
 '0416','0387','0426','0407','0418','0399','0421','0397','0413','0410','0388','0408','0401','0406','0429','0423','0417','0395','0422','0425','0404','0394']

os.makedirs('oasis/testing')
os.makedirs('oasis/training')
os.makedirs('oasis/validation')

for img in all_data:
    index = img.split('_')[-2]
    if index in testing:
        shutil.move(img, 'oasis/testing')
    elif index in validation:
        shutil.move(img, 'oasis/validation')
    else:
        shutil.move(img, 'oasis/training')