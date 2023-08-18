# Mask Projection Generator

## Overview

This Python project is designed to generate mask projections from image data. The primary functionality includes the application of masks to images and the simulation of the propagation of these masked images based on various parameters. 

## Prerequisites

- Python 3.11 or higher
- Libraries: NumPy, SciPy, PyTorch, Pillow (PIL), Torch
- Dataset: Mnist(binary version is used in this project, please see test_label_binary and train_label_binary)

## Run it
- Open cmd, navigate to the script's directory, and then:
- <pre> python generator_mask_projection.py<pre>
- or you can create a Jupyter notebook under the same directory as generator_mask_projection.py and other .py files  then:
- <pre> run generator_mask_projection.py<pre>
- It will produce a directory called "size_[]m_inter_[]_lambda_dis_[]m_feq_[]hz_water" under the current directory, where [] depends on your parameter setup in generator_mask_projection.py
- <pre> dist = 0.15
    masking_size = 2
    masking_p = [10, 20]
    start_time = time.time()
    
    cwater = 1500
    freq = 200000
    numspfreqcomp = 300
    numOfImg = 499
    lambda_ = cwater / freq
    h = 0.03 (distance between each acoustic source or pixel in our picture)
    distance = 0.15 (distance between receiver and acoustic source)
    inter_d = h / 32 / lambda_

    name = 'size_'+str(h)+'m_inter_'+str(inter_d)+'_lambda_dis_'+str(distance)+'m_feq_'+str(freq)+'hz_water' <pre>


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

- [Chu Ma Research Group](<[link_to_the_group_page](https://ma.ece.wisc.edu/)>)
- [Mnist Dataset Source](<[link_to_the_dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)https://www.kaggle.com/datasets/hojjatk/mnist-dataset>)
