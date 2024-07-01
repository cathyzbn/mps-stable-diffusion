
import asyncio
import json
from ray import serve
from request_generator import measure_throughput_and_latency, update_batch_size, update_model_id, update_num_workers, update_arrival_rate
from server import get_entrypoint
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
    model_id = config['server']['model_id']
    batch_size = config['server']['batch_size']
    num_workers = config['server']['num_workers']

f.close()


arrival_rates_unsaturated = {
    "stabilityai/sd-turbo": {
        1: 0.64,
        2: 0.64,
        4: 0.39,
        8: 0.34,
        16: 0.324
    },
    "stabilityai/stable-diffusion-xl-base-1.0": {
        1: 1.4,
        2: 1.4,
        4: 0.79,
        8: 0.51,
        16: 0.472
    }
}

arrival_rates_saturated = {
    "stabilityai/sd-turbo": {
        1: 0.64,
        2: 0.41,
        4: 0.35,
        8: 0.32,
    },
    "stabilityai/stable-diffusion-xl-base-1.0": {
        1: 1.4,
        2: 0.89,
        4: 0.61,
        8: 0.5,
    }
}

for model_id in ["stabilityai/sd-turbo", "stabilityai/stable-diffusion-xl-base-1.0"]:
    update_model_id(model_id)
    for bs in [1, 2, 4, 8]:
        for num_workers in [2, 4, 8]:
            update_num_workers(num_workers)
            if bs * num_workers > 8:
                continue
            if model_id == "stabilityai/stable-diffusion-xl-base-1.0" and num_workers > 4:
                continue
            batch_size = bs
            update_batch_size(batch_size)
            eff_batch_size = batch_size * num_workers

            # arrival_rate = arrival_rates_saturated[model_id][eff_batch_size]
            # update_arrival_rate(arrival_rate)

            print(f"Model: {model_id}, Batch Size: {batch_size}, Arrival Rate: {arrival_rate}")
            serve.run(get_entrypoint())
            asyncio.run(measure_throughput_and_latency(url, num_requests, output_file, arrival_rate))

# print(f"Model: {model_id}, Batch Size: {batch_size}, Arrival Rate: {arrival_rate}, Num Worker: {num_workers}")
# serve.run(get_entrypoint())
# asyncio.run(measure_throughput_and_latency(url, num_requests, output_file, arrival_rate))