import tensorflow as tf
import glob
import os
import pandas as pd
from sklearn.model_selection import train_test_split

#load file list of images and their corresponding labels
def load_file_list_and_labels(directory,path_to_csv):
    
    age_df=pd.read_csv(path_to_csv)
    age_df['path'] = age_df['id'].map(lambda x: os.path.join(directory, 
                                                          '{}.png'.format(x)))

    age_df["gender"]=age_df['male'].map(lambda x: 0 if x==True else 1)//assign binary labels for males and females
    file_list=age_df['path'].values.tolist()
    labels=age_df['gender'].values.tolist()

    return file_list,labels

#load images
def load_image_and_labels(image_filename,label):
    image_bytes=tf.read_file(image_filename)
    pixels=decode_image(image_bytes)
    return image_filename,pixels,label

def decode_image(img_bytes):
    pixels = tf.image.decode_image(img_bytes,channels=1)
    return tf.cast(pixels, tf.uint8)

#resize images to 228*228  for the input of our network
def resize_image(filename,image,label):
    image.set_shape([2044,1514,1])
    pixels=tf.image.resize_images(image,size=[228,228],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return filename,pixels,label
#augment the data I did the following augmentation
#1 rotate images
#2 flip images
def random_orientation(name,image, label):
    # This function will output boxes x1, y1, x2, y2 in the standard orientation where x1 <= x2 and y1 <= y2
    rnd = tf.random_uniform([], 0, 9, tf.int32)
    img = image_tile

    def f0(): return tf.image.rot90(img, k=0)
    def f1(): return tf.image.rot90(img, k=1)
    def f2(): return tf.image.rot90(img, k=2)
    def f3(): return tf.image.rot90(img, k=3)
    def f4(): return tf.image.rot90(tf.image.flip_left_right(img), k=0)
    def f5(): return tf.image.rot90(tf.image.flip_left_right(img), k=1)
    def f6(): return tf.image.rot90(tf.image.flip_left_right(img), k=2)
    def f7(): return tf.image.rot90(tf.image.flip_left_right(img), k=3)
    def f8(): return tf.contrib.image.rotate(img,angles=20)
    image_tile = tf.case({tf.equal(rnd, 0): f0,
                                tf.equal(rnd, 1): f1,
                                tf.equal(rnd, 2): f2,
                                tf.equal(rnd, 3): f3,
                                tf.equal(rnd, 4): f4,
                                tf.equal(rnd, 5): f5,
                                tf.equal(rnd, 6): f6,
                                tf.equal(rnd, 7): f7,
                                tf.equal(rnd, 8): f8})

    return name,image_tile,label
#input function to feed the data to our model
def train_input_fn(filelist,labels,batch_size,num_epochs,resize=True,rnd_orientation):
    dataset= tf.data.Dataset.from_tensor_slices((tf.constant(file_list), tf.constant(labels)))
    dataset=dataset.map(load_image_and_labels)
    if resize==True:
       dataset=dataset.map(resize_image)
    if rnd_orientation==True#for augmentation
       dataset=dataset.map(random_orientation)
    if shuffle==True:
       dataset=dataset.shuffle(1000)
       
    dataset=dataset.batch(batch_size).repeat(num_epochs)
        
    iterator=dataset.make_one_shot_iterator()
    name,features,labels=iterator.get_next()
    return {'images':features,'names':name},labels

def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # images are 228*228 pixels, and have one color channel
  features=tf.cast(features['images'],tf.float32)
  
  
  input_layer=tf.reshape(features,(-1,228,228,1),name="input_layer")
  # Convolutional Layer #1
  # Computes  features using a 11x11 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 228, 228, 1]
  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=96,
      kernel_size=[11, 11],
      strides=4,
      name="convolution1",
      activation=tf.nn.relu)

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3], strides=1,padding="same")

  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=256,
      kernel_size=[5, 5],
      strides=2,
      name="convolution2",
      activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[3, 3], strides=1,padding="same")
  
  # Convolutional Layer #3
  conv3 = tf.layers.conv2d(
      inputs=pool2,
      filters=384,
      kernel_size=[3, 3],
      strides=2,
      name="convolution3",
      activation=tf.nn.relu)
  
  # Convolutional Layer #4
  conv4 = tf.layers.conv2d(
      inputs=conv3,
      filters=384,
      kernel_size=[3, 3],
      name="convolution4",
      activation=tf.nn.relu,padding="same")
  
  # Convolutional Layer #5
  conv5 = tf.layers.conv2d(
      inputs=conv4,
      filters=256,
      kernel_size=[3, 3],
      name="convolution5",
      activation=tf.nn.relu,padding='same')

  flat = tf.reshape(conv5, [-1,12*12*256])
  
  dense1 = tf.layers.dense(inputs=flat, units=4096, activation=tf.nn.relu,name="dense1",)
  #add dropout regularization
  dropout1 = tf.layers.dropout(
      inputs=dense1, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)
    
  dense2 = tf.layers.dense(inputs=dropout1, units=4096, activation=tf.nn.relu,name="dense2")

  dropout2 = tf.layers.dropout(
      inputs=dense1, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)
    
    
  # Logits Layer
  logits = tf.layers.dense(inputs=dropout2, units=2,name=logits)#shape=(batch_size,2)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1,name="predicted_classes"),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor",name="probabilities")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits,name="loss")

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(argv):
   parser=argparse.ArgumentParser()
   parser.add_argument("--path_to_csv", help="path to your csv file")
   parser.add_argument("--data_dir", help="path to your data folder")
    
   args = parser.parse_args()
   arguments = args.__dict__
   path_to_csv= arguments["path_to_csv"]
   data_dir= arguments["data_dir"]
   filelist, labels= load_file_list_and_labels(data_dir, path_to_csv)
   Xtrain, Xtest, ytrain, ytest = train_test_split(filelist, labels, test_size=0.25, random_state=47)
   batch_size=10
   num_epochs=200
   rnd_orientation=True
   train_ip=lambda:train_input_fn(Xtrain, Xtest, batch_size, num_epochs, resize=True, rnd_orientation):

   estimator = tf.estimator.Estimator(model_fn=cnn_model_fn,
                                    model_dir='train_dir',
                                    params=hparams)

   estimator.train(input_fn=train_ip, max_steps=200000)
   eval_ip= train_input_fn(ytrain, ytest, batch_size=10, num_epochs=1, resize=True, rnd_orientation=False)
   eval_res=estimator.evaluate(input_fn=eval_ip)
   print("accuracy=",eval_res)

if __name__ == '__main__':
     main(sys.argv)

