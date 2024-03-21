# attention_regulation

##TODO
We are working on refactoring our messy code. Code will be released soon :)


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

