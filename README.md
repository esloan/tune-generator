# tune-generator
Playing with using tensorflow RNNs to generate traditional Irish Music

To generate tunes, run RNNgenerate.py or RNNnoFixedLen.py in a Python3 environment that uses tensorflow.
 * RNNgenerate.py generates a single reel, with enforced AABB structure where each part is 8 measures.
 * RNNnoFixedLen.py generates some amount of music in ABC notation based on a character cut off with no enforced structure.
 * RNNnoTrain.py was a shortcut for me to load an already trained model and generate tunes using it - very similar to RNNgenerate.py

Data is from http://cobb.ece.wisc.edu/irish/Tunebook.html
Much code/inspiration taken from the tensorflow RNN tutorial here: https://www.tensorflow.org/tutorials/sequences/text_generation

This code is a work in progress, for more useful work on algorithmic tune generation, see https://github.com/IraKorshunova/folk-rnn
