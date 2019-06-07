# Emotion-Recognition
This repository builds a neural network to detect the emotions such as anger, disgust, fear, happiness, sadness, surprise, neutral and comfort from facial expressions. 

Initial part of the training is done with a deep CNN on the University of Denver's AffectNet database together with a thousand of manually labelled marathon pictures. Furthermore, five years of Marathon demographics information such as age, gender, pace etc. is scraped from Chicago and London Marathons from 2014 to 2018. The latent vectors of the pre-trained deep CNN is concatanated with the demographics information to pedict the marathon runners' emotional state.

Documentation for this project is in progrees but a web app of the project can be found here: https://nostradamusproject.herokuapp.com/
