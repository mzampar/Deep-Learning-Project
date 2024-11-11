Repository for the DL project.

The aim is to perform Nowcasting on the evolution of the rainfall in FVG through a CNN.

The images were downloaded from the Protezione Civile FVG DataBase "https://monitor.protezionecivile.fvg.it/api".

## The model

This model is inspired by the paper Convolutional LSTM Network: A Machine Learning
Approach for Precipitation Nowcasting 
https://arxiv.org/pdf/1506.04214

![](./conv_lstm.png)

In the first figure we can see a ConvLSTM cell, the only difference form a LSTM cell is that the input and the hidden state are passed through a convolution. 

![](./model.png)

In the second figure we can see the structure of the model, which is not completely clear to me.

A summary of the structure:

Input frames

ConvLSTM 1

Input = Input_frames

Output: Hidden_states, Context

Initial_hidden_state = zeros

Initial_context = zeros

At this point the hidden states are passed to the second ConvLSTM, but how are the context and the first hidden state intialised?

ConvLSTM 2

Input = Hidden states 1

Output: Hidden_states, Context

Initial_hidden_state = ?

Initial_context = ?

From the figure, it is not clear what is the input of the third ConvLSTM. 

About the initialisation of the states, the paper says: "the initial states and cell outputs of the forecasting network are copied from the last state of the encoding network" and from the figure we can see that ConvLSTM 1 -> ConvLSTM 3, and ConvLSTM 2 -> ConvLSTM 4.

This is clear beacuse the last context and the last hidden state of the encoding network, which are the hidden representation of the last input frame, are used to predict the next frame (its hidden state). From this first predicted frame we update the context and predict the hidden state of the second frame and so on.

It is not clear what are the inputs of the third ConvLSTM, maybe simply there aren't?

I would say that the input is the last frame of the image, beacuse the network 3 is initialised with the context of network 1. Network 1 knows how to extract the hidden state and update the context from an image, not from an image's hidden state.

But an lstm works with a sequence... what is the next input of the third ConvLSTM?

It doesn't make sense to give all the frames because they have already been processed.

ConvLSTM 3

Input = ?

Output: Hidden_states, Context

Initial_hidden_state = last hidden state 1

Initial_context = last context 1

The fourth ConvLSTM is intialised with the hidden state and the context of the last frame of the second ConvLSTM. What is the purpose of the second ConvLSTM? It is to extract the context of the hidden states of frames. 

ConvLSTM 4

Input = Hidden_states 3 (?)

Output: Hidden_states, Context

Initial_hidden_state = last hidden state 2

Initial_context = last context 2


About the prediction, the authors say "we concatenate all the states in the forecasting network and feed them into a 1x1 convolutional layer to generate the final prediction."

So it seems that we leverage the hidden states of both ConvLSTM 3 4 to do the predictions, whose length is the same of the initial sequence.



IT

Predicted frames


___________________

code from: https://github.com/chengtan9907/OpenSTL/blob/OpenSTL-Lightning/openstl/models/convlstm_model.py
