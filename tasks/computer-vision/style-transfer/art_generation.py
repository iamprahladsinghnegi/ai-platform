# Importing packages
import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import mlflow
import mlflow.tensorflow
import warnings
from matplotlib.pyplot import imshow
from PIL import Image
from nst_utils import *
from compute_content_cost import *
from compute_style_cost import *
from total_cost import *
from model_nn import *
from gram_matrix import *


# Clear
# Previous Tensorflow Session
tf.keras.backend.clear_session()
tf.logging.set_verbosity(tf.logging.ERROR)


class Default:
    CONTENT_IMAGE = 'images/test.jpg'
    STYLE_IMAGE = 'images/style.jpg'
    STYLE_LAYERS = [('conv1_1', 0.2), ('conv2_1', 0.2), ('conv3_1', 0.2), ('conv4_1', 0.2), ('conv5_1', 0.2)]
    ALPHA = 10
    BETA = 40
    CONTENT_LAYER = '5_2'
    STYLE_LNAME = ['1_1','2_1','3_1','4_1','5_1']
    STYLE_LCOFF = [0.2,0.2,0.2,0.2,0.2]
    ITERATIONS = 1000



if __name__ == "__main__":
    
    warnings.filterwarnings("ignore")
    def_val = Default()

    alpha = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] != '-NA' else def_val.ALPHA # content_image weight
    beta = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] != '-NA' else def_val.BETA # style_image weight
    content_image = str(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] != '-NA' else def_val.CONTENT_IMAGE
    style_image = str(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] != '-NA' else def_val.STYLE_IMAGE
    content_layer = 'conv{}'.format(sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] != '-NA' else def_val.CONTENT_LAYER)
    style_lname = sys.argv[6].split(',') if len(sys.argv) > 6 and sys.argv[6] != '-NA' else def_val.STYLE_LNAME
    style_lcoff = sys.argv[7].split(',') if len(sys.argv) > 7 and sys.argv[7] != '-NA' else def_val.STYLE_LCOFF
    iterations = int(sys.argv[8]) if len(sys.argv) > 8 and sys.argv[8] != '-NA' else def_val.ITERATIONS

    if len(style_lname) == len(style_lcoff):
        style_layers= [('conv{}'.format(style_lname[i]),float(style_lcoff[i])) for i in range(len(style_lname))]
    else:
        print("length of style_lname did not match with style_lcoff")
        inn = input('DO you want to use default STYLE_LAYERS (Y/N) : ')
        if inn.upper() == 'Y':
            style_layers = def_val.STYLE_LAYERS
        else:
            exit()
            


    with mlflow.start_run() as run:
        
        mlflow.log_param("alpha", alpha) 
        mlflow.log_param("beta", beta)
        mlflow.log_param("content_layer", content_layer)
        mlflow.log_param("style_lname", style_lname)
        mlflow.log_param("style_lcoff", style_lcoff)
        mlflow.log_param("style_layers", style_layers)
        mlflow.log_param("iterations", iterations)
    
        # Reset the graph
        tf.reset_default_graph()

        # Start interactive session
        sess = tf.InteractiveSession()

        # Loading content_image
        content_image = scipy.misc.imread(content_image)

        # Resizing the conent_image 
        content_image = scipy.misc.imresize(content_image, (300, 400))
        content_image = reshape_and_normalize_image(content_image)

        # Loading style_image
        style_image = scipy.misc.imread(style_image)

        #Resizing the style_image
        style_image = scipy.misc.imresize(style_image, (300, 400))
        style_image = reshape_and_normalize_image(style_image)

        # Initializing the generated_image as a noisy image created from the content_image
        generated_image = generate_noise_image(content_image)
        #imshow(generated_image[0])

        # Loading the VGG-19 model
        model = load_vgg_model("model/imagenet-vgg-verydeep-19.mat")

        # Assign the content_image to be the input of the VGG model.  
        sess.run(model['input'].assign(content_image))

        # Select the output tensor of content_layer
        out = model[content_layer]

        # Set a_C to be the hidden layer activation from the layer we have selected
        a_C = sess.run(out)

        # Set a_G to be the hidden layer activation from same layer. Here, a_G references model[content_layer] 
        # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
        # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
        a_G = out

        # Compute the content cost
        J_content = compute_content_cost(a_C, a_G)

        # Assign the input of the model to be the style_image
        sess.run(model['input'].assign(style_image))

        # Compute the style cost
        J_style = compute_style_cost(model, style_layers, sess)

        # Totla cost
        J = total_cost(J_content, J_style, alpha , beta)
      
        # define optimizer
        optimizer = tf.train.AdamOptimizer(2.0)

        # define train_step 
        train_step = optimizer.minimize(J)

        cost = [J, J_content, J_style]
      
        output_image, Jt, Jc, Js = model_nn(model, sess, generated_image, train_step, cost, iterations)

        mlflow.log_metric("Total_cost", Jt)
        mlflow.log_metric("J_content", Jc)
        mlflow.log_metric("J_style", Js)
        mlflow.log_artifact("output/generated_image.jpg")
        mlflow.log_artifact("output/log_cost.txt")

