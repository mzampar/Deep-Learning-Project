### Data 

The dataset used to perform the training was created autonomously from the Protezione Civile FVG DataBase "https://monitor.protezionecivile.fvg.it/api". It contains images of the SRI (surface rainfall intensity) from the Fossalon (Grado) radar.

Images were:

1. Downloaded
2. Cropped
3. Resized and turned in grey scale using torch.transforms
4. Saved as tensors

The notebooks used to preprocess the images are available in the `data` folder.

A useful dataframe was created. It contains:

1. Image ID
2. Date and Time
3. Amount of rain
4. Rain Category
5. Sequence ID

The amount of rain was estimated using the number of grey pixels contained in each image. A grey pixel is a pixel in which no rain is detected.

The seqeunce ID was established by looking at the difference in time between 2 contiguous images: if the difference is more than 10 minutes, a new sequence is created, to not have a too big discontinuity between 2 images.

Too short sequences of low-rainy-sequences were discarded.