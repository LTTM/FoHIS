
# 🌫️ FoHIS: Simulating Foggy and Hazy Images

## 🛠️ Instructions

Set your custom parameters in `parameters.py`, then use the function `fog` to generate foggy images. Inputs:

- 🖼️ `rgb path`: Path to the input image
- 🗺️ `depth path`: Path to the depth image
- 💾 `out path`: Path to the output image
- 💡 `luminance reduction factor`: Optional, controls how much the luminance is reduced (default: 0)
- 🎨 `saturation reduction factor`: Optional, controls how much the saturation is reduced (default: 0)
- 🌈 `apply color scheme`: Optional, if True applies a premade color scheme that enhances the fog effect (default: False)
- 🌫️ `min-depth`: Optional, sets the minimum depth for the fog effect (default: 1)

To run as standalone code, use:

```bash
python fog.py --rgb <path> [--depth <path> [--out <path>] [--reduce-lum <value>] [--reduce-sat <value>] [--color-scheme] [--min-depth <value>]
```

## 🙏 Acknowledgements

This repository is a fork of the one made by Ning Zhang, Lin Zhang, and Zaixi Cheng for their paper [Towards Simulating Foggy and Hazy Images and Evaluating their Authenticity](https://link.springer.com/chapter/10.1007/978-3-319-70090-8_42). The original repository can be found [here](https://github.com/noahzn/FoHIS).

This specific fork was created by:

- [Matteo Caligiuri](https://github.com/matteocali),
- [Francesco Barbato](https://github.com/barbafrank).
