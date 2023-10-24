# KaggleX__Mentorship_Project
This repo contains the files and details for my project for Kaggle Mentorship.

It is an end to end Machine Learning project with the aim of building and deploying a Machine Learning model that predicts the species and generates details/information of bird when given a bird image.

## Project Description
The project is about building a Machine Learning model that takes an image of a bird as input and then gives the bird species and details/information about the bird as output.

It involeves two Machine Learning models;
1. Vision Transformer Model: This model is fine-tuned to classify bird images. It gives the bird species as output.
2. Large Language Model (LLM): The output of the Vision Transformer model is fed into this model as a prompt to give the details about the bird as an output.

The models are then deployed with flask as a web application.

## Data
The data used is the BIRDS 525 SPECIES- IMAGE CLASSIFICATION dataset from Kaggle. 
Link to the data https://www.kaggle.com/datasets/gpiosenka/100-bird-species

## Models
The Vision Transformer model used is the https://huggingface.co/google/vit-base-patch16-224-in21k 

The Large Language used is the https://huggingface.co/pankajmathur/orca_mini_3b


