import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests

correct_label = tf.placeholder(tf.float32)
learning_rate = tf.placeholder(tf.float32)
keep_prob = tf.placeholder(tf.float32)

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    with sess.as_default(): 
        tf.saved_model.loader.load(sess, ['vgg16'], vgg_path)
        graph = sess.graph
        image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
        keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
        layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
        layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
        layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return image_input, keep_prob, layer3_out, layer4_out, layer7_out

tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    kernel_initializer = tf.truncated_normal_initializer(stddev = 0.01)

	# Transform the Fully Connected Layer output of layer 7, 4 and 3 to a 1*1 Convolution layer
	# l1_* is misleading as it is the transformed 7th layer from VGG16, did not find good naming convention
    l1_conv_1x1 = tf.layers.conv2d(inputs=vgg_layer3_out, filters=num_classes, kernel_size=(1,1), strides=(1,1), kernel_initializer=kernel_initializer)
    l2_conv_1x1 = tf.layers.conv2d(inputs=vgg_layer4_out, filters=num_classes, kernel_size=(1,1), strides=(1,1), kernel_initializer=kernel_initializer)
    l3_conv_1x1 = tf.layers.conv2d(inputs=vgg_layer7_out, filters=num_classes, kernel_size=(1,1), strides=(1,1), kernel_initializer=kernel_initializer)

    # Decoder Layer with upsampling and skipped connections
    # Upsampling l1_conv_1x1
    l4_decoder = tf.layers.conv2d_transpose(inputs=l3_conv_1x1, filters=num_classes, kernel_size=(4,4), strides=(2,2),padding='same', kernel_initializer=kernel_initializer)
    # Skip connections from VGG16 layer 4
    l5_decoder = tf.add(l4_decoder, l2_conv_1x1)
    # Upsampling l5_decoder
    l6_decoder = tf.layers.conv2d_transpose(inputs=l5_decoder, filters=num_classes, kernel_size=(4,4), strides=(2,2), padding='same', kernel_initializer=kernel_initializer)
    # Skip connections from VGG16 layer 3
    l7_decoder = tf.add(l6_decoder, l1_conv_1x1)
    # Upsampling l7_decoder
    output = tf.layers.conv2d_transpose(inputs=l7_decoder, filters=num_classes, kernel_size=(16,16), strides=(8,8), padding='same', kernel_initializer=kernel_initializer)

    return output

tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # Reshape 4D tensors to 2D ones (reach row being a pixel and each column a class)
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label = tf.reshape(correct_label, (-1, num_classes))

    # Define cost function (which aims at lower the cross_entropy_loss)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label)) 

    # Define optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    return logits, optimizer, cost

tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    KEEP_PROBABILITY = 0.75
    LEARNING_RATE = 0.0001

    sufficient_cost = 1
    with sess.as_default():
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            index = 0
            for images, labels in get_batches_fn(batch_size):
                index += 1
                _, loss = sess.run([train_op, cross_entropy_loss], feed_dict = { 
                    input_image: images,
                    correct_label: labels,
                    keep_prob: KEEP_PROBABILITY,
                    learning_rate: LEARNING_RATE 
                })
                print("Batch: ", '%04d | ' % (index), "cost =", "{:.9f}".format(loss))
            print("Epoch: ", '%04d | ' % (epoch+1), "cost =", "{:.9f}".format(loss))
tests.test_train_nn(train_nn)


def run():
    
    EPOCHS = 5
    BATCH_SIZE = 8
    
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # Build NN using load_vgg, layers, and optimize function
        image_input, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)

        # Get the output of the model
        nn_last_layer = layers(layer3_out, layer4_out, layer7_out, num_classes)

        # Train NN using the train_nn function
        logits, optimizer, cost = optimize(nn_last_layer, correct_label, learning_rate, num_classes)
        train_nn(sess, EPOCHS, BATCH_SIZE, get_batches_fn, optimizer, cost, image_input, correct_label, keep_prob, learning_rate)

        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, image_input)


if __name__ == '__main__':
    run()
