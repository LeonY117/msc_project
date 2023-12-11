# Leon's MSc Thesis: Efficient Bayesian Neural Networks for Outdoor Semantic Scene Understanding Tasks in Robotics

The full thesis can be viewed <a href="https://www.leonyao.net/pdfs/MSc_Thesis_LeonYao.pdf">here</a>.

## Problem

<!-- intro image -->
Deep neural networks often suffer from overconfidence and slow computation. My thesis focused on two aspects:

1. making networks more efficient
2. perform fast Bayesian inference on these networks

### Dataset
Camvid dataset: `367`, `101`, and `233` train, val, test images respectively, trained and tested with resolution `480 x 360`

```
Dataset format:

```

## Methodology

A segmentation network was built using inverted residual blocks.

<!-- network design -->

A novel Bayesian inference technique is proposed using stochastic depths.

<!-- Bayesian inference: stochastic depth -->


## Results

### Network architecture variations

* *no-skip*: remove skip connections in the encoder-decoder framework


### Uncertainties 
