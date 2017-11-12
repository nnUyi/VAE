# VAE
A implement of VAE in paper "Auto-Encoding Variational Bayes"

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
  
    $ python main.py
  
  (4)testing
    
   - Anyway, in this repo, testing processing is taken during training, It samples the training results to the sample(named sample) directory, and stores session in the checkpoint(named checkpoint) directory.

# Contacts

  Email:computerscienceyyz@163.com, Thank you for contacting if you find something wrong or if you have some problems!!!
