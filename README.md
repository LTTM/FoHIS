
# 🌫️ FoHIS: Simulating Foggy and Hazy Images

## 🛠️ Instructions

Set your custom parameters in `parameters.py`, then use the function `fog` to generate foggy images. Inputs:

- 🖼️ `rgb path`: Path to the input image
- 🗺️ `depth path`: Path to the depth image
- 💾 `out path`: Path to the output image
- 💡 `luminance reduction factor`: Optional, controls how much the luminance is reduced (default: 0)
- 🎨 `saturation reduction factor`: Optional, controls how much the saturation is reduced (default: 0)
- 🌫️ `depth flattening`: Optional, whether to apply depth flattening (default: False)
- 🔍 `depth multiplier`: Optional, multiplier for fog intensity (default: None)

To run as standalone code, use:

```bash
python fog.py --rgb <path> [--depth <path> [--out <path>] [--reduce_lum <value>] [--reduce_sat <value>] [--depth_flattening] [--depth_multiplier <value>]
```

## 🙏 Acknowledgements

This repository is a fork of the one made by Ning Zhang, Lin Zhang, and Zaixi Cheng for their paper [Towards Simulating Foggy and Hazy Images and Evaluating their Authenticity](https://link.springer.com/chapter/10.1007/978-3-319-70090-8_42). The original repository can be found [here](https://github.com/noahzn/FoHIS).

This specific fork was created and maintained by [Matteo Caligiuri](https://github.com/matteocali).
