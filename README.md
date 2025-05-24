Codebase for 'A Racing Dataset and Baseline Model for Track Detection in Autonomous Racing'[https://arxiv.org/html/2502.14068v1] currently under review. The dataset is available at https://www.kaggle.com/datasets/shreya64/roratrack-dataset.

Training:

main.py trains the model. Specify weight and result paths in the code. 

Testing:

test.py tests the model. Code requires trained weights before testing. Specify trained weight path and result paths in the code. For reference, trained RaceGAN model weights are available at weights/GAN/. These can be used to regenerate results in the paper. 

Dataloader:

