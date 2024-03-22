# Attention Regulation

This repository contains the PyTorch inference code for the paper "Enhancing Semantic Fidelity in Text-to-Image Synthesis: Attention Regulation in Diffusion Models". Paper link: [arXiv](https://arxiv.org/abs/2403.06381)


## Environment Setup
Clone this repository
```bash
git clone https://github.com/YaNgZhAnG-V5/attention_regulation.git
cd attention_regulation
```
Install the required dependencies with the supported versions 
```bash
pip install -r requirements.txt
```

## Usage
We provide a script ([txt2img.py](txt2img.py)) for inference. You can use it to generate images from text using our Attention Regulation approach.

Example usage:
```bash
python txt2img.py --prompt "A painting of a bag and a apple" --target "bag apple"
```

The full list of options is as follows:
```
usage: txt2img.py [-h] --prompt PROMPT [--target TARGET] [--workdir WORKDIR] [--cuda-id CUDA_ID] [-n N] [-s STEPS] [--guidance-scale GUIDANCE_SCALE] [--seed SEED] [--edit-steps EDIT_STEPS] [--layers [LAYERS ...]]
                  [--pipeline-id PIPELINE_ID] [--scheduler SCHEDULER]

options:
  -h, --help            show this help message and exit
  --prompt PROMPT       Prompt
  --target TARGET       Target phrase for editing, separated by space
  --workdir WORKDIR     Working directory
  --cuda-id CUDA_ID     CUDA device id
  -n N                  Number of images to generate per prompt
  -s STEPS, --steps STEPS
                        Number of inference steps
  --guidance-scale GUIDANCE_SCALE
                        Guidance scale
  --seed SEED           Random seed
  --edit-steps EDIT_STEPS
                        Number of edit steps
  --layers [LAYERS ...]
                        Layers to edit. Select from: ['down_blocks.0','down_blocks.1','down_blocks.2', 'mid_block', 'up_blocks.1', 'up_blocks.2', 'up_blocks.3']
  --pipeline-id PIPELINE_ID
                        Pipeline ID from Diffusers. We support SD 1.4 SD 1.5 SD 2 and SD 2.5
  --scheduler SCHEDULER
                        Scheduler to use
```


## Acknowledgements
If you find our work useful for your work, please consider citing our paper:
```
@misc{zhang2024enhancing,
      title={Enhancing Semantic Fidelity in Text-to-Image Synthesis: Attention Regulation in Diffusion Models}, 
      author={Yang Zhang and Teoh Tze Tzun and Lim Wei Hern and Tiviatis Sim and Kenji Kawaguchi},
      year={2024},
      eprint={2403.06381},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
