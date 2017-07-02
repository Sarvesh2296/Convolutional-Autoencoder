import tensorflow as tf
import numpy as np

def load_image():
  filename_queue = tf.train.string_input_producer(['./Coco_subset_4/COCO_train2014_000000000110.jpg']) #  list of files to read


  reader = tf.WholeFileReader()
  key, value = reader.read(filename_queue)

  my_img = tf.image.decode_jpeg(value) # use png or jpg decoder based on your files.


  init_op = tf.global_variables_initializer()
  with tf.Session() as sess:
    sess.run(init_op)

    # Start populating the filename queue.

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(1): #length of your filename list
      image = my_img.eval() 

    print(image.shape)
    image = tf.image.resize_images(image,[256,256])
    print(image.shape)
    print(type(image))
    image = tf.reshape(image, [1,256,256,3])
    print(image.shape)
    coord.request_stop()
    coord.join(threads)
    return image

load_image()

def print_h():
  print "Hello"