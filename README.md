# Classification-of-X-Ray-Images
Let's see whether Neural Networks can find whether the X-Ray image is of male or female

## Training
Download the data from kaggle https://www.kaggle.com/kmader/rsna-bone-age/ and unzip it, it will have a train folder containiing images and a .csv file containing labels
```
git clone https://github.com/abajaj945/Classification-of-X-Ray-Images.git
cd Classification-of-X-Ray-Images
train.py --path_to_csv path/toyour/csv file --data_dir path/to/directory/containing/training/images
```
Without data augmentation the model achieves a acccuracy of 70%

With data augmentation the model achieves a accuracy of 80%

The model can improved by

1.Increased image resolution for training

2.Using a deeper network

3. Tuning Hyperparameters
