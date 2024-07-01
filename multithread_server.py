from io import BytesIO
from typing import List
from fastapi import FastAPI
from fastapi.responses import Response, JSONResponse
import torch
import json
import time
from concurrent.futures import ThreadPoolExecutor

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
        model_id = config['server']['model_id']
        num_workers = config['server']['num_workers']

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

            image, server_start_time, end_time = await self.handle.generate.remote(prompt)
            file_stream = BytesIO()
            image.save(file_stream, "PNG")
            return JSONResponse(content={"server_start_time": server_start_time, "end_time": end_time})

    @serve.deployment(
        ray_actor_options={"num_gpus": (1)},
    )
    class StableDiffusionV2:
        def __init__(self):
            from diffusers import DPMSolverMultistepScheduler, AutoPipelineForText2Image
            import torch

            self.pipes = []
            for _ in range(num_workers):
                pipe = AutoPipelineForText2Image.from_pretrained(
                    model_id, torch_dtype=torch.float16, variant="fp16"
                )
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                    pipe.scheduler.config
                )
                pipe = pipe.to("cuda")
                self.pipes.append(pipe)
            self.executor = ThreadPoolExecutor(max_workers=num_workers) 
            self.cur_idx = 0

        def generate_images(self, prompts, model_idx):
            pipe = self.pipes[model_idx]
            with torch.autocast("cuda"):
                images = pipe(prompts, height=img_size, width=img_size, num_inference_steps=16).images
                return images

        @serve.batch(max_batch_size=batch_size, batch_wait_timeout_s=batch_wait_timeout_s)
        async def generate(self, prompts: List[str]):
            assert len(prompts), "prompt parameter cannot be empty"
            server_start_time = time.time()

            future = self.executor.submit(self.generate_images, prompts, self.cur_idx) 
            self.cur_idx = (self.cur_idx + 1) % num_workers
            results = future.result()
            return [(r, server_start_time, time.time()) for r in results]

    entrypoint = APIIngress.bind(StableDiffusionV2.bind())
    return entrypoint
