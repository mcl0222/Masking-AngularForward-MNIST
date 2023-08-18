import os
import numpy as np
import time
import torch
from PIL import Image
from masking import masking
from angular_forward import AngularForward


def main():
    dist = 0.15
    masking_size = 2
    masking_p = [10, 20]
    start_time = time.time()
    
    cwater = 1500
    freq = 200000
    numspfreqcomp = 300
    numOfImg = 499
    lambda_ = cwater / freq
    h = 0.03
    distance = dist
    inter_d = h / 32 / lambda_

    name = 'size_'+str(h)+'m_inter_'+str(inter_d)+'_lambda_dis_'+str(distance)+'m_feq_'+str(freq)+'hz_water'
    print(name)
    # Get the directory of the current Python script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct full paths to the "test_label_binary" and "train_label_binary" folders (input folders)
    test_label_binary_path = os.path.join(script_dir, 'test_label_binary')
    train_label_binary_path = os.path.join(script_dir, 'train_label_binary')

    # Construct full paths for where you want to save the processed test and train images (output folders)
    # save_test_directory = os.path.join(script_dir, 'saved_test_images')
    # save_train_directory = os.path.join(script_dir, 'saved_train_images')
    if not os.path.exists(name):
        os.makedirs(name)
    for i in range(10):
        os.makedirs(os.path.join(name, str(i)))
    
    for idx, p in enumerate(masking_p):
        for b in range(10):
            pathtest = os.path.join(test_label_binary_path, str(b))
            savepathtest = os.path.join(script_dir, name, str(b), 'test_' + str(b))
            
            pathtrain = os.path.join(train_label_binary_path, str(b))
            savepathtrain = os.path.join(script_dir, name, str(b), 'train_' + str(b))
            
            os.makedirs(os.path.join(name, str(b), 'train_' + str(b)))
            os.makedirs(os.path.join(name, str(b), 'test_' + str(b)))
            
            for c, filename1 in enumerate(os.listdir(pathtest)):
                print(f"Currently working on: Dataset: [TEST]   size {h}m: image -- [{b}] -- {c} -- {c}|{numOfImg} ---  time: {time.time()-start_time}s!")
                
                filepath = os.path.join(pathtest, filename1)
                ima = np.array(Image.open(filepath))
                imaB = masking(32, 32, p, ima, masking_size)
                
                angf = AngularForward(numspfreqcomp, freq, cwater, h, distance)
                imaC = angf.process(torch.tensor(imaB))
                
                im = Image.fromarray((imaC.numpy() * 255).astype(np.uint8))
                print("save path test:",savepathtest)
                im.save(os.path.join(savepathtest, filename1))
            
            for c, filename2 in enumerate(os.listdir(pathtrain)):
                print(f"Currently working on: Dataset: [Training]   size {h}m: image -- [{b}] -- {c} -- {c}|{numOfImg} ---  time: {time.time()-start_time}s!")
                
                filepath = os.path.join(pathtrain, filename2)
                ima = np.array(Image.open(filepath))
                imaB = masking(32, 32, p, ima, masking_size)
                
                angf = AngularForward(numspfreqcomp, freq, cwater, h, distance)
                imaC = angf.process(torch.tensor(imaB))
                
                im = Image.fromarray((imaC.numpy() * 255).astype(np.uint8))
                im.save(os.path.join(savepathtrain, filename2))
        
    print("FINISHED! Yahoo! total time:", time.time() - start_time, 's')

if __name__ == '__main__':
    main()
