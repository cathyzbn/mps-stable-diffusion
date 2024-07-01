
import asyncio
import json
from ray import serve
from request_generator import measure_throughput_and_latency, update_batch_size, update_model_id
from multithread_server import get_entrypoint
import pandas as pd
import numpy as np

np.random.seed(0)

config_file = 'config.json'

with open(config_file) as f:
    config = json.load(f)
    num_requests = config['request_generator']['num_requests']
    arrival_rate = config['request_generator']['arrival_rate']
    output_file = config['request_generator']['output_file']
    prompt = config['request_generator']['prompt']
    url = config['request_generator']['url']
    batch_size = config['server']['batch_size']
    num_workers = config['server']['num_workers']

f.close()


for model_id in ["stabilityai/stable-diffusion-2", "stabilityai/sd-turbo", "stabilityai/stable-diffusion-xl-base-1.0"]:
    update_model_id(model_id)
    for bs in [2]:
        batch_size = bs
        update_batch_size(batch_size)
        
        print(f"Model: {model_id}, Batch Size: {batch_size}, Arrival Rate: {arrival_rate}")
        serve.run(get_entrypoint())
        asyncio.run(measure_throughput_and_latency(url, num_requests, output_file, arrival_rate, arrival_times=arrival_times[model_id]))

serve.run(get_entrypoint())
asyncio.run(measure_throughput_and_latency(url, num_requests, output_file, arrival_rate))