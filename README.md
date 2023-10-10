# Towards Simulating Foggy and Hazy Images and Evaluating their Authenticity

## Instructions

Set the custom parameters in the `parameters.py` file than use the function `fog` to generate the fog. It takes as inputs:

- `input_img path`: path to the input image,
- `depth_img path`: path to the depth image,
- `output_img path`: path to the output image,
- `luminance reduction factor`: optional parameter that define how much the luminance of the image is reduced. Default value is 0,

If you want to run this as a standalone code simply run the followinmg command:

```python
python fog.py --input_img <path> --depth_img <path> --output_img <path> --reduce_lum
```

## Aknowledgements

This repository is a fork of the one made by Ning Zhang, Lin Zhang, and Zaixi Cheng for their paper [Towards Simulating Foggy and Hazy Images and Evaluating their Authenticity](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123480154.pdf). The original repository can be found [here](https://github.com/zhengziqiang/ForkGAN). 

This specific fork was created and maintained by [Matteo Caligiuri](https://github.com/matteocali).
