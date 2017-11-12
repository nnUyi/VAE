# VAE
  
  A implement of VAE in paper "Auto-Encoding Variational Bayes". In this repo, we assume that our data is a continus type, so Gaussian Distribution is taken and applied in this repo. If you have a binary data type, Bernoulli Distribution is a good idea, then what you need to change is to use crossy entrophy in marginal loss instead of reconstruction loss. More details can be seen in this [vae-tutorial](https://home.zhaw.ch/~dueo/bbs/files/vae.pdf)

# Requirement
  - tensorflow 1.3.0

  - python2.7 or python3.6

  - numpy 1.13.*

  - scipy 0.17.0~ 0.19.1
  
# Usage
  (1)download this repo to your own directory and enter it
  
    $ git clone https://github.com/nnUyi/VAE.git
    $ cd VAE
    
  (2)download celebA dataset and store it in the data directory(directory named data)
      
   - celebA datasets is cropping into 64*64 size with *.png or *jpg format, this repo read image format data as input.
      
  (3)training
  
    $ python main.py --is_training=True
  
  (4)testing
    
   - Anyway, in this repo, testing processing is taken during training, It samples the training results to the sample(named sample) directory, and stores session in the checkpoint(named checkpoint) directory.

# Experimental Result
   <p align='center'><img src='train_29_0900.png'></p>
  
  This repo is trained with celebA, after 30 epoches, we can clear see the result above.
  
# Contacts

  Email:computerscienceyyz@163.com, Thank you for contacting if you find something wrong or if you have some problems!!!
