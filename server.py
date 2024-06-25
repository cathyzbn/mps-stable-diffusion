from io import BytesIO
from typing import List
from fastapi import FastAPI
from fastapi.responses import Response, JSONResponse
import torch
import json
import time

from ray import serve
from ray.serve.handle import DeploymentHandle


app = FastAPI()
config_file = 'config.json'

def get_entrypoint():
    with open(config_file) as f:
        config = json.load(f)
        batch_size = config['server']['batch_size']
        batch_wait_timeout_s = config['server']['batch_wait_timeout_s']
        img_size = config['server']['img_size']
        num_workers = config['server']['num_workers']
        model_id = config['server']['model_id']

    @serve.deployment(num_replicas=1)
    @serve.ingress(app)
    class APIIngress:
        def __init__(self, diffusion_model_handle: DeploymentHandle) -> None:
            self.handle = diffusion_model_handle

        @app.get(
            "/imagine",
            responses={200: {"content": {"image/png": {}}}},
            response_class=Response,
        )
        async def generate(self, prompt: str):
            assert len(prompt), "prompt parameter cannot be empty"

            # image, end_time = await self.handle.generate.remote(prompt)
            image, end_time = await self.handle.generate.remote(prompt)
            file_stream = BytesIO()
            image.save(file_stream, "PNG")
            # return Response(content=file_stream.getvalue(), media_type="image/png")
            # return Response(content=file_stream.getvalue(), headers=str(end_time), media_type="image/png")
            return JSONResponse(content={"end_time": end_time})


    @serve.deployment(
        ray_actor_options={"num_gpus": (1.0/num_workers)},
        autoscaling_config={"min_replicas": num_workers, "max_replicas": num_workers},
    )
    class StableDiffusionV2:
        def __init__(self):
            from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline, AutoPipelineForText2Image, StableDiffusionXLPipeline
            from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
            import torch

            if model_id == "stabilityai/stable-diffusion-2":
                self.pipe = StableDiffusionPipeline.from_pretrained(
                    model_id, revision="fp16", torch_dtype=torch.float16
                )
            elif model_id == "stabilityai/sd-turbo":
                self.pipe = AutoPipelineForText2Image.from_pretrained(
                    model_id, torch_dtype=torch.float16, variant="fp16"
                )
            elif model_id == "stabilityai/stable-diffusion-xl-base-1.0":
                self.pipe = StableDiffusionXLPipeline.from_pretrained(
                    model_id, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
                )
            else:
                raise ValueError(f"Invalid model_id: {model_id}")
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config
            )
            # self.pipe.enable_xformers_memory_efficient_attention(
            #     attention_op=MemoryEfficientAttentionFlashAttentionOp
            # )
            # self.pipe.vae.enable_xformers_memory_efficient_attention(attention_op=None)
            self.pipe = self.pipe.to("cuda")

        @serve.batch(max_batch_size=batch_size, batch_wait_timeout_s=batch_wait_timeout_s)
        async def generate(self, prompts: List[str]):
            assert len(prompts), "prompt parameter cannot be empty"

            with torch.autocast("cuda"):
                images = self.pipe(prompts, height=img_size, width=img_size).images
                outputs = [(im, time.time()) for im in images]
                return outputs
                

    entrypoint = APIIngress.bind(StableDiffusionV2.bind())
    return entrypoint
