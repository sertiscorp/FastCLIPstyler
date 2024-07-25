#FastCLIPstyler

## Setup
Tested with `python 3.8.13` on Ubuntu 18.04.6 LTS.

```
conda create -n fastclipstyler python=3.8
pip install -r requirements.txt
conda install -c anaconda git
pip install git+https://github.com/openai/CLIP.git
```

## Inference


In order to run inference with the attached trained model, please run
```
python inference.py
```

This will run the inference with the trained FastCLIPstyler model.
To change the text prompt/content image, please change the `test_prompts` variable in `inference.py`.

To run the EdgeCLIPstyler model, please run change the `text_encoder` feild in the `params` class to from `fastclipstyler` to `edgeclipstyler`.


