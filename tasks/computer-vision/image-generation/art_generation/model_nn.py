# Importing packages
import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from nst_utils import *
import numpy as np
import tensorflow as tf





def model_nn(model, sess, input_image, train_step, cost, num_iterations):
    
    # Initialize global variables
    sess.run(tf.global_variables_initializer())
    
    # Run the noisy input image (initial generated image) through the model.
    sess.run(model['input'].assign(input_image))

    # Creating a log_cost file ( keep log of every 20 iteration)
    if not os.path.exists("output"): os.mkdir("output") # Check or create output directory
    file = open("output/log_cost.txt","w+")
    
    for i in range(1,num_iterations+1):
    
        # Run the session on the train_step to minimize the total cost
        sess.run(train_step)
        
        # Compute the generated image by running the session on the current model['input']
        generated_image = sess.run(model['input'])


        # Print every 10 iteration.
        if i%10 == 0:
            J, J_content, J_style = cost
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))

            # writing in log_cost file
            file.write("Iteration " + str(i) + " :\n")
            file.write("total cost = " + str(Jt) + "\n")
            file.write("content cost = " + str(Jc) + "\n")
            file.write("style cost = " + str(Js) + "\n")            
            
            # save current generated image in the "/output" directory
            save_image("output/" + str(i) + ".png", generated_image)
            
    
    # save last generated image
    save_image('output/generated_image.jpg', generated_image)
    file.write("\n Final Iteration : \n")
    file.write("total cost = " + str(Jt) + "\n")
    file.write("content cost = " + str(Jc) + "\n")
    file.write("style cost = " + str(Js) + "\n")
    file.close()
    
    return generated_image, Jt, Jc, Js
