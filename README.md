# Rock-Paper-Scissors
Standard Rock Paper Scissors implementation done using python and Deep Learning. Model has seen to reach upto 99% accuracy. Total accuracy will vary based on the amount of data available. 

# Data Creation
Data is loaded using create_training_data() function.
Initially, Synthetic data and some kaggle datasets were used but were found to be inaccurate due to difference in realtime conditions.
Hence, in case a realtime data isn't available or you just wanna do it in home for yourself and those around you, it can be generated using the function *data_creation.py*
This can further be improved by data augmentation (May get added in future versions)

# Models
I've tried Alexnet and Resnet here.
Resnet to me worked best.
If more information is needed, cheggout the papers.
I modified the final layer of both Alexnet and resnet to work with the current data and classes.
Resnet achieved a test accuracy of 99% with a training loss of 0.5

# Putting it together
Driver.py is to test your model
Once you make your model, run it through main.py and haffun

# Future Versions
1. Might include an AI Algorithm to increase your competition (AI Labs help me out)
2. Image Augmentation might be implemented to support lesser amount of factory data.
3. Synthetic data to real data conversion using GANS. (Will happen once I learn GANs)
