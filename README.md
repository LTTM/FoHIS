
# 🌫️ FoHIS: Simulating Foggy and Hazy Images

## 🛠️ Instructions

Set your custom parameters in `parameters.py`, then use the function `fog` to generate foggy images. Inputs:

- 🖼️ `input_img path`: Path to the input image
- 🗺️ `depth_img path`: Path to the depth image
- 💾 `output_img path`: Path to the output image
- 💡 `luminance reduction factor`: Optional, controls how much the luminance is reduced (default: 0)

To run as standalone code, use:

```bash
python fog.py --input_img <path> --depth_img <path> --output_img <path> --reduce_lum <value>
```

## 🙏 Acknowledgements

This repository is a fork of the one made by Ning Zhang, Lin Zhang, and Zaixi Cheng for their paper [Towards Simulating Foggy and Hazy Images and Evaluating their Authenticity](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123480154.pdf). The original repository can be found [here](https://github.com/zhengziqiang/ForkGAN).

This specific fork was created and maintained by [Matteo Caligiuri](https://github.com/matteocali).
