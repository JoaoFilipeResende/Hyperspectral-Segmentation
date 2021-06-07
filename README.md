# Segmentation of Hyperspectral Images
Hyperspectral image segmentation using the Hyko dataset, which is a collection of labeled hyperspectral images taken from an area scan camera mounted on a moving car, to train classical (kNN) and neural-network based machine learning models.

Example of the per-pixel labeled data:
![Hyko Labeled Example](https://github.com/rereee3/Hyperspectral_Segmentation/blob/master/Figures/vis_img_ex.png)

## Proposed Solutions:

- kNN Algorithm:
  - Classifies pixels according to their spectra (spectrum used as feature vector);
	- For each type of image (VIS and NIR), two models with k=3 and k=5 were tested;
	- For both VIS and NIR, 80% of the available images were used as training samples and 20% as test samples.
	- VIS:
	  - 130 images for training (16840200 pixels);
	  - 32 images for testing (4145280 pixels);
	- NIR:
	  - 62 images for training (5400076 pixels);
	  - 16 images for testing (1393568 pixels);

- Neural Network: Full Image
![ANN Full Image](https://github.com/rereee3/Hyperspectral_Segmentation/blob/master/Figures/ann_full.png)

  - Takes an entire hyperspectral image as input and attempts to classify all pixels at once;
  - Network is fully convolutional, comprising 3D convolutional layers followed by 3D max pooling layers on the downsampling path and the inverse operations on the upsampling path;
  - The network’s final layer is a single-filter 3D convolutional layer;
  - Since the network uses 3D layers, the output is also 3D;
  - For each type of image (VIS and NIR), two models with 8 filters and 16 filters on the first layer were tested (number of filters double according to depth);
  - For both VIS and NIR, 70% of the available images were used for training, 10% for validation and 20% for testing.
	- VIS:
      - 114 images for training;
	  - 16 images for validation;
	  - 32 images for testing;
	- NIR:
	  - 55 images for training;
	  - 7 images for validation;
	  - 16 images for testing;



- Neural Network: Patches
![ANN Patches](https://github.com/rereee3/Hyperspectral_Segmentation/blob/master/Figures/ann_patches.png)

	- Takes a patch of NxNxL from a hyperspectral image as input and attempts to classify the center pixel;
	  - State-of-the-art technique;
	- For each type of image (VIS and NIR), two models with N=3 and N=5 were tested;
	- For both VIS and NIR, 70% of the available images were used for training, 10% for validation and 20% for testing.
	- VIS:
	  - 114 images for training (14767560 pixels);
      - 16 images for validation (2072640 pixels);
	  - 32 images for testing (4145280 pixels);
	- NIR:
	  - 55 images for training (4790390 pixels);
	  - 7 images for validation (609686 pixels);
	  - 16 images for testing (1393568 pixels);

![ANN Patches](https://github.com/rereee3/Hyperspectral_Segmentation/blob/master/Figures/ann_patches.png)

## Results:

- Example of the outputs of different methods (NIR)
![ANN Patches](https://github.com/rereee3/Hyperspectral_Segmentation/blob/master/Figures/results_nir.png)

- Example of the outputs of different methods (VIS)
![ANN Patches](https://github.com/rereee3/Hyperspectral_Segmentation/blob/master/Figures/results_vis.png)
