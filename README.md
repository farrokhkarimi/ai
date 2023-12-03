**[farrokhkarimi.github.io](https://farrokhkarimi.github.io/)**

# ai
Artificial Intelligence Fundamental Concepts

# Fundamental Concepts
[Natural Neural Network and Artificial Neural Network Fundamental Concepts](https://docs.google.com/presentation/d/1Q0x1RXbXiUs7bOq1qfQryeZMMfv2GFwcdy0UVR4dNzQ/) [[3D Brain]](http://www.g2conline.org/3dbrain/)

# Image Processing
[Image Processing with OpenCV](https://github.com/farrokhkarimi/OpenCV) [[Colab notebook link]](https://colab.research.google.com/github/farrokhkarimi/OpenCV/blob/master/Getting_Started_with_OpenCV.ipynb)

# TensorFlow and PyTorch Installation
[TensorFlow installation guide](https://www.tensorflow.org/install)  
[PyTorch installation guide](https://pytorch.org/get-started/locally/)  
[Google Colab](https://colab.research.google.com/)

# Deep Learning
* [TensorFlow 2 quickstart](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/quickstart/beginner.ipynb)  
  * [Built-in small datasets](https://keras.io/api/datasets/)  
  * [Activation functions](https://keras.io/api/layers/activations/)  
  * [Loss functions](https://keras.io/api/losses/)  
  * [Optimizers](https://keras.io/api/optimizers/)  
  * [Metrics](https://keras.io/api/metrics/)  
  * [Architectures](https://keras.io/api/applications/)  
  * [Callbacks](https://keras.io/api/callbacks/)  
* [TensorFlow Datasets](https://colab.research.google.com/github/tensorflow/datasets/blob/master/docs/overview.ipynb) [[Catalog]](https://www.tensorflow.org/datasets/catalog/overview)
* [[Teachable Machine V1]](https://teachablemachine.withgoogle.com/v1/) [[Teachable Machine V2]](https://teachablemachine.withgoogle.com/train/)
* [Convolutional Neural Network (CNN)](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/images/cnn.ipynb) [[Paper]](https://arxiv.org/pdf/1511.08458.pdf) [[Explainer]](https://poloclub.github.io/cnn-explainer/)  
* [Image classification, [Data Augmentation], [Batch Normalization], [Overfitting], and [Dropout]](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/images/classification.ipynb)  
* [Transfer learning and fine-tuning](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/images/transfer_learning.ipynb)
* [Autoencoders](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/generative/autoencoder.ipynb) [[Paper]](https://arxiv.org/pdf/2003.05991.pdf)  
* [Variational Autoencoder (VAE)](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/generative/cvae.ipynb) [[Paper]](https://arxiv.org/pdf/1906.02691.pdf)  
* [MusicVAE](https://colab.research.google.com/github/magenta/magenta-demos/blob/master/colab-notebooks/MusicVAE.ipynb) [[Paper]](https://arxiv.org/pdf/1803.05428.pdf) [[Reference]](https://magenta.tensorflow.org/music-vae) 
* [Generative Adversarial Network (GAN)](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/generative/dcgan.ipynb) [[Paper1]](https://arxiv.org/pdf/1406.2661.pdf) [[Paper2]](https://arxiv.org/pdf/1511.06434.pdf) [[Paper3]](https://arxiv.org/pdf/1701.00160.pdf) [[Scribble Diffusion]](https://scribblediffusion.com/) [[ChatGPT]](https://openai.com/blog/chatgpt/) [[GPTZero]](https://gptzero.me/)
* [Pix2Pix (Image-to-image translation with a [Conditional GAN])](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/generative/pix2pix.ipynb) [[Paper1]](https://arxiv.org/pdf/1411.1784.pdf) [[Paper2]](https://arxiv.org/pdf/1611.07004.pdf) [[Reference]](https://phillipi.github.io/pix2pix/) [[Demo]](https://affinelayer.com/pixsrv/)   
* [Image Segmentation with U-Net](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/images/segmentation.ipynb) [[Paper]](https://arxiv.org/pdf/1505.04597.pdf)
* [Recurrent Neural Networks (RNN)](https://colab.research.google.com/github/tensorflow/docs/blob/snapshot-keras/site/en/guide/keras/rnn.ipynb) [[Paper1]](https://arxiv.org/pdf/1808.03314.pdf) [[Paper2]](https://arxiv.org/ftp/arxiv/papers/1701/1701.05923.pdf) [[Textbook]](http://dprogrammer.org/rnn-lstm-gru) 

<!--  

-->

# Appendix
**To plot the model:**  
```python3
tf.keras.utils.plot_model(model, rankdir="TD", show_shapes=True)
```

**Parameters:**  
FC: (the previous layer number of nodes * the next layer number of nodes) + the next layer number of biases  
CNNs: (kernel size (w*h) * number of channels * number of filters) + number of biases  

**Conv Feature Map Size:**  
The size of the convoluted matrix is given by C=((I-F+2P)/S)+1, where C is the size of the Convoluted matrix, I is the size of the input matrix, F is the size of the filter matrix and P is the padding applied to the input matrix.

# Author
https://farrokhkarimi.github.io/
