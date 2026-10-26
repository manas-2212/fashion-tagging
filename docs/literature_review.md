# Literature Review (Phase 1)

## 1. Fashion-MNIST as a drop-in replacement for MNIST [1]

Xiao, Rasul and Vollgraf introduce Fashion-MNIST: 70,000 grayscale images of fashion products, 28x28 pixels, in 10 categories with 7,000 images each. The official split is 60,000 training and 10,000 test images. The authors present it as a direct drop-in replacement for MNIST. It has the same image size, data format and split structure, so existing MNIST code needs essentially only a different download location. Their motivation is that MNIST is popular because it is small, simply encoded and supported out of the box by common libraries, but it is an easy task. Fashion-MNIST is meant to keep those practical advantages while being a more challenging classification problem.

On construction, the images are front-view thumbnails of 70,000 unique products from the Zalando online shop; white products were left out because they contrast poorly with the background. A seven-step conversion pipeline turns each thumbnail into a 28x28 grayscale image (trimming, resizing the longest edge to 28, sharpening, centring on the canvas, inverting intensities, 8-bit grayscale). Labels are the products' silhouette codes, assigned manually by in-house fashion experts and reviewed by a separate team. The training set takes 6,000 randomly selected examples per class. The files use the MNIST format and are stored sorted by label, so the authors leave shuffling to the user. The paper's benchmark table covers scikit-learn classifiers, with test accuracy averaged over five runs with shuffled training data. It contains no convolutional network.

## 2. LeCun et al. and learned features [2]

LeCun, Bottou, Bengio and Haffner review methods for handwritten character recognition and compare them on a standard handwritten digit task. Their abstract argues that multilayer networks trained by back-propagation can learn complex decision surfaces on high-dimensional patterns, such as handwritten characters, with minimal preprocessing. They report that convolutional networks, which are designed for the variability of 2D shapes, outperform all the other techniques compared in that study. The paper also introduces graph transformer networks for training multi-module document recognition systems; this is not used in our project. The point we take from it is the case for learned over hand-designed features: let the model learn the spatial features instead of engineering them.

## 3. Link to our project

Our question is whether a small CNN, which learns spatial features, recognises similar clothing classes better than classical models on fixed representations: raw pixels (no feature design) and HOG (hand-designed gradient features). Fashion-MNIST [1] supplies the data and the official 60,000/10,000 split. Because the files are sorted by label, our 54,000/6,000 train/validation split is stratified and seeded (seed 42) instead of taking the first rows. LeCun et al. [2] supply the motivation for the learned-feature side of the comparison.

| | Fixed representation (pixels, HOG) | Learned representation (CNN) |
|---|---|---|
| Features | Fixed before training (identity, or HOG with chosen parameters) | Learned from the training data |
| Classifier | Logistic regression, k-NN, random forest, RBF-SVM | Convolutional layers plus a linear output |
| Main risk | The representation may miss useful structure | More parameters and a more costly training procedure |

## 4. What these papers did not do

Neither paper ran our experiments. [1] does not evaluate HOG features or a CNN on Fashion-MNIST. [2] studies handwritten digits and documents, not clothing images. So neither paper tells us which method wins here. We make no claim about their reported numbers, and our result is whatever we measure under our protocol.

## References

[1] H. Xiao, K. Rasul and R. Vollgraf, "Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms," arXiv:1708.07747, 2017. https://arxiv.org/abs/1708.07747

[2] Y. LeCun, L. Bottou, Y. Bengio and P. Haffner, "Gradient-based learning applied to document recognition," Proc. IEEE, vol. 86, no. 11, pp. 2278-2324, Nov. 1998. DOI 10.1109/5.726791
