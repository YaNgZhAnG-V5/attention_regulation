import sys

sys.path.append("..")
from attention_regulation.editor import StableDiffusionEditor

import torch
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True, help="Prompt")
    parser.add_argument(
        "--target", type=str, help="Target phrase for editing, separated by space"
    )
    parser.add_argument("--workdir", type=str, default=".", help="Working directory")

    parser.add_argument("--cuda-id", type=int, default=0, help="CUDA device id")
    parser.add_argument(
        "-n", type=int, default=10, help="Number of images to generate per prompt"
    )
    parser.add_argument(
        "-s", "--steps", type=int, default=50, help="Number of inference steps"
    )
    parser.add_argument(
        "--guidance-scale", type=float, default=7.5, help="Guidance scale"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--edit-steps", type=int, default=25, help="Number of edit steps"
    )
    parser.add_argument(
        "--layers",
        nargs="*",
        type=str,
        default=["up_blocks.1", "down_blocks.2"],
        help="Layers to edit. Select from: ['down_blocks.0','down_blocks.1','down_blocks.2', 'mid_block', 'up_blocks.1', 'up_blocks.2', 'up_blocks.3']",
    )
    parser.add_argument(
        "--pipeline-id",
        type=str,
        default="stabilityai/stable-diffusion-2-base",
        help="Pipeline ID from Diffusers. We support SD 1.4 SD 1.5 SD 2 and SD 2.5",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="DPMSolverMultistepScheduler",
        help="Scheduler to use",
    )
    args = parser.parse_args()

    seed = args.seed
    overall_workdir = args.workdir
    num_images_per_prompt = args.n
    pmt = args.prompt
    target_phrase = args.target

    sd_editor = StableDiffusionEditor(
        args.pipeline_id,
        args.scheduler,
        args.cuda_id,
        num_images_per_prompt,
        args.steps,
        args.edit_steps,
        args.layers,
        args.guidance_scale,
    )

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    workdir = f"{overall_workdir}/AttentionRegulation/{pmt}"
    os.makedirs(workdir, exist_ok=True)
    images = sd_editor.txt2img(pmt, target_phrase)

    # save image
    for i, image in enumerate(images):
        image.save(f"{workdir}/img_{i}.png")


if __name__ == "__main__":
    main()
