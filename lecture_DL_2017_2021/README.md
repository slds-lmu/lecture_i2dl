# Deep Learning Lecture

The repository latex-math is used. Please read its ReadMe here: https://github.com/compstat-lmu/latex-math

Moodle: https://moodle.lmu.de/course/view.php?id=10446

Teaching Plan: https://docs.google.com/spreadsheets/d/1giiEyqfJccavqARpUsyAyyViDhFwByz8PqCBC76R2QU/edit#gid=0

Revision/ Changes: https://docs.google.com/document/d/1k6JmHEg74szZf4A03EypOXlBX0qoFh8jFkIzEOGzCTg/edit#heading=h.q5t9zzmxt2yo


# General rules

Please observe the following rules when creating and editing the lecture and exercise slides:

## Setup
1. Clone this repository
2. Clone the latex-math repository into the main directory of this repository
3. Navigate to a folder where the slideset is contained, e.g. 2020/01-introduction
4. If there is Makefile in the folder: do make -f "Makefile" unless render slides by knitr::knit2pdf("slides.Rnw")


# Structure

Topics will be added or remade, the normal ones are already in the slides:


1. Introduction, Overview, and a breif history of deep learning 

2. Deep Forward Neural Network, Gradient Descent, Backpro, Hard ware and Software

3. Regularization of NNs, early stopping  
	
4. Dropout and challenge in optimization

5. Advance in Optimization

6. Activation Function and Initialization

7. Convolutional Neural Network, Variant CNN, Applications

8. Modern CNN and Overview of some applications

9. Recurrent Neural Network

10. Modern RNN and applications

11. Deep Unsupervised Learning 

12. Autoencoders, AE Regularization and Variant

13. Manifold Learning

14. Deep Generative Models, VAE, GANs
 
# Math and formula
1. Math environments within a text line are created by $ environment, separate equation lines are created by $$ environment

2. The abbreviations defined in the header file should always be used within the code for simplification purposes

3. The repo latex-math is used. Please read the corresponding ReadMe: https://github.com/compstat-lmu/latex-math

# Material Deep Learning 


## Extra material to have a look at:

* [ Alex J. Smola (2020): *Dive into Deep Learning*](https://d2l.ai/index.html) (An interactive deep learning book with code, math, and discussions Provides NumPy/MXNet, PyTorch, and TensorFlow implementations)(free HTML version)
* [Goodfellow, Bengio, Courville (2016): *Deep Learning*](http://www.deeplearningbook.org/) (free HTML version)
* [Awesome Deep Learning](https://github.com/ChristosChristofidis/awesome-deep-learning)
* [Andrej Karpathy blog](http://karpathy.github.io/)
* [Coursera Kurs "Neural Networks for Machine Learning"](https://www.coursera.org/learn/neural-networks#syllabus)
* [Youtube Talk von Geoff Hinton "Recent Developments in Deep Learning"](https://www.youtube.com/watch?v=vShMxxqtDDs)
* [Practical Deep Learning For Coders - contains many detailed python notebooks on how to implement different DL architectures](http://course.fast.ai/index.html)
* [The Matrix Calculus You Need For Deep Learning](http://parrt.cs.usfca.edu/doc/matrix-calculus/index.html) 
* [Tensorflow Neural Network Playground](https://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.14139&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false)
* [A Weird Introduction to Deep Learning](https://towardsdatascience.com/a-weird-introduction-to-deep-learning-7828803693b0)
* [Deep Learning Achievements Over the Past Year](https://blog.statsbot.co/deep-learning-achievements-4c563e034257)
* [Scalable Deep Learning (Talk)](https://determined.ai/blog/talk-scalable-dl/)
* [Deep Learning Resources](https://sebastianraschka.com/deep-learning-resources.html)

## Good Websites to have a look 
 * [distill.pub](https://distill.pub/): in-depth explanations of important concepts, worth checking out periodically for new material


## Optimization / Training of NNs:

 * [Why Momentum Really Works](https://distill.pub/2017/momentum/)
 * [Adam -- latest trends in deep learning optimization](https://towardsdatascience.com/adam-latest-trends-in-deep-learning-optimization-6be9a291375c)
 * [Overview of Gradient Descent Optimization Algorithms](https://ruder.io/optimizing-gradient-descent/)
 * [Yes you should understand backprop](https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b)
 * [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/)

## Regularization:

* [Regularization for Deep Learning: A Taxonomy](https://arxiv.org/pdf/1710.10686.pdf)

## CNNs:

* [The Sobel and Laplacian Edge Detectors](http://aishack.in/tutorials/sobel-laplacian-edge-detectors/)
* [Keras Blog: How convolutional neural networks see the world](https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html)
* [The 9 Deep Learning Papers You Need To Know About](https://adeshpande3.github.io/The-9-Deep-Learning-Papers-You-Need-To-Know-About.html)
* [Python based visualization repo for CNNs](https://github.com/HarisIqbal88/PlotNeuralNet)
* [Computing Receptive Fields of Convolutional Neural Networks](https://distill.pub/2019/computing-receptive-fields/)
* [Deconvolution and Checkerboard Artifacts](https://distill.pub/2016/deconv-checkerboard/)
* [Understanding Convolution in Deep Learning](http://timdettmers.com/2015/03/26/convolution-deep-learning/)    
* [What do we learn from region based object detectors (Faster R-CNN, R-FCN, FPN)]
* [How Convolutional Neural Networks see the World](https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html)
* [Attention in Neural Networks and How to Use It](http://akosiorek.github.io/ml/2017/10/14/visual-attention.html)
* [Neural Networks - A Systematic Introduction (FU Berlin)](https://page.mi.fu-berlin.de/rojas/neural/neuron.pdf)
* [Deep Learning - The Straight Dope (contains notebooks designed to teach deep learning)](https://gluon.mxnet.io/)
* [Computing Receptive Fields of Convolutional Neural Networks](https://distill.pub/2019/computing-receptive-fields/)
* Stanford: Convolutional Neural Networks for Visual Recognition
	* [Videos](http://cs231n.stanford.edu/)
	* [Assignements und notes](http://cs231n.github.io/) 
* 

### Autoencoders

* [PCA](http://www.cs.cmu.edu/~guestrin/Class/15781/slides/pca-mdps-annotated.pdf)
* [Introducing Variational Autoencoders (in Prose and Code)](https://blog.fastforwardlabs.com/2016/08/12/introducing-variational-autoencoders-in-prose-and.html)

### Variational Autoencoders

* [A Tutorial on Variational Autoencoders](https://arxiv.org/pdf/1606.05908.pdf)



### Reinforcement Learning

* [Statistical Reinforcement Learning (Lecture)](http://nanjiang.cs.illinois.edu/cs598/)
* [Practical Reinforcement Learning (Course)](https://github.com/yandexdataschool/Practical_RL)



## LSTMs:

* [Understanding LSTM and its diagrams](https://medium.com/mlreview/understanding-lstm-and-its-diagrams-37e2f46f1714)
* [The most comprehensive yet simple and fun RNN/LSTM tutorial on the Internet.](https://ayearofai.com/rohan-lenny-3-recurrent-neural-networks-10300100899b)


### Hyperparameter Optimization / Neural Architecture Search / etc:

* [Using Machine Learning to Explore Neural Network Architecture](https://ai.googleblog.com/2017/05/using-machine-learning-to-explore.html)


### Software / Languages / etch

* [R vs Python: Image Classification with Keras](https://towardsdatascience.com/r-vs-python-image-classification-with-keras-1fa99a8fef9b)
* H20 related stuff: 
	* [Repository that contains the H2O presentation for Trevor Hastie and Rob Tibshirani's Statistical Learning and Data Mining IV course in Washington, DC on October 19, 2016.](https://github.com/ledell/sldm4-h2o)
	* [Deep Learning with H2O - PDF/Book](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/booklets/DeepLearningBooklet.pdf)


### Nice Demos and Vizualisations

* [Deep Traffic](https://selfdrivingcars.mit.edu/deeptraffic/)


### Material for Exercises

* [Neural networks Exercises (Part-1)](https://www.r-bloggers.com/neural-networks-exercises-part-1/)


### External material

1. other
* PyData-2017-TF_TFS: Slides for Spark + Tensorflow + Notebooks

