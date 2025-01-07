''' 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np
import streamlit as st 

model = load_model('C:\\Users\\Sharooz Ali\\Desktop\\Local Disk(D)\\VegFruit_Classification_CNN\\Image_classify.keras')
data_category = ['apple',
 'banana',
 'beetroot',
 'bell pepper',
 'cabbage',
 'capsicum',
 'carrot',
 'cauliflower',
 'chilli pepper',
 'corn',
 'cucumber',
 'eggplant',
 'garlic',
 'ginger',
 'grapes',
 'jalepeno',
 'kiwi',
 'lemon',
 'lettuce',
 'mango',
 'onion',
 'orange',
 'paprika',
 'pear',
 'peas',
 'pineapple',
 'pomegranate',
 'potato',
 'raddish',
 'soy beans',
 'spinach',
 'sweetcorn',
 'sweetpotato',
 'tomato',
 'turnip',
 'watermelon']

img_height = 180
img_width = 180
st.header('Image Classification Model')
image = st.text_input('Enter Image Name','Apple.jpg')

image_load = tf.keras.utils.load_img(image, target_size=(img_height, img_width))
img_arr = tf.keras.utils.array_to_img(image_load)
img_bat = tf.expand_dims(img_arr,0)

predict = model.predict(img_bat)
score = tf.nn.softmax(predict)
st.image(image,width=200)
st.write('Veg/Fruit in image is '+ data_category[np.argmax(score)])
st.write('With accuracy of ' + str(np.max(score)*100))
'''

import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import streamlit as st 

# Load the pre-trained model
model = load_model('Image_classify.keras')

# Define data categories
data_category = ['apple',
 'banana',
 'beetroot',
 'bell pepper',
 'cabbage',
 'capsicum',
 'carrot',
 'cauliflower',
 'chilli pepper',
 'corn',
 'cucumber',
 'eggplant',
 'garlic',
 'ginger',
 'grapes',
 'jalepeno',
 'kiwi',
 'lemon',
 'lettuce',
 'mango',
 'onion',
 'orange',
 'paprika',
 'pear',
 'peas',
 'pineapple',
 'pomegranate',
 'potato',
 'raddish',
 'soy beans',
 'spinach',
 'sweetcorn',
 'sweetpotato',
 'tomato',
 'turnip',
 'watermelon']

# Set image dimensions
img_height = 180
img_width = 180

# Streamlit UI
st.header('Image Classification Model')
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the image
    image_load = tf.keras.utils.load_img(uploaded_file, target_size=(img_height, img_width))
    img_arr = tf.keras.utils.img_to_array(image_load)
    img_bat = tf.expand_dims(img_arr, 0)

    # Make a prediction
    predict = model.predict(img_bat)
    score = tf.nn.softmax(predict)

    # Display the image and prediction results
    st.image(image_load, width=200)
    st.write('Veg/Fruit in image is: **' + data_category[np.argmax(score)] + '**')
    st.write('With accuracy of: **' + str(np.max(score) * 100)[:5] + '%**')
else:
    st.write("Please upload an image file.")
