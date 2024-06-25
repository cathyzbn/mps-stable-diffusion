import subprocess
import time
import asyncio
import json
from ray import serve
from request_generator import measure_throughput_and_latency, update_batch_size, update_model_id
from server import get_entrypoint

config_file = 'config.json'

with open(config_file) as f:
    config = json.load(f)
    num_requests = config['request_generator']['num_requests']
    arrival_rate = config['request_generator']['arrival_rate']
    output_file = config['request_generator']['output_file']
    prompt = config['request_generator']['prompt']
    url = config['request_generator']['url']

f.close()

# "stabilityai/sd-turbo", "stabilityai/stable-diffusion-xl-base-1.0"
# for model_id in ["stabilityai/stable-diffusion-2", , ]:
for model_id in [ "stabilityai/stable-diffusion-xl-base-1.0"]:
    update_model_id(model_id)
    for bs in [16]:
        batch_size = bs
        update_batch_size(batch_size)
        serve.run(get_entrypoint())
        asyncio.run(measure_throughput_and_latency(url, num_requests, arrival_rate, output_file))

# serve.run(get_entrypoint())
# asyncio.run(measure_throughput_and_latency(url, num_requests, arrival_rate, output_file))