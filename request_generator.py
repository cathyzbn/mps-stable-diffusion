
import aiohttp
import asyncio
import time
import numpy as np
import pandas as pd
import json


np.random.seed(0)

config_file = 'config.json'
with open(config_file) as f:
    config = json.load(f)
    num_workers = config['server']['num_workers']
    
    num_requests = config['request_generator']['num_requests']
    arrival_rate = config['request_generator']['arrival_rate']

    compute_type = config['request_generator']['compute_type']
    output_file = config['request_generator']['output_file']
    prompt = config['request_generator']['prompt']
    url = config['request_generator']['url']
f.close()


input = "%20".join(prompt.split(" "))


async def send_get_request(session, url, input):
    start_time = time.time()
    request = session.get(f"{url}?prompt={input}")
    task = asyncio.create_task(request)
    
    #await the request to ensure it's sent
    await asyncio.sleep(0)
    return task, start_time

async def measure_throughput_and_latency(url, num_requests, arrival_rate, output_file):
    total_time = 0
    successful_requests = 0

    async with aiohttp.ClientSession() as session:
        tasks = []
        sts = []
        latencies = []

        start_time = time.time()
        
        for i in range(num_requests):
            task, st = await send_get_request(session, url, input)

            tasks.append(task)
            sts.append(st)

            if arrival_rate != 0:
                inter_arrival_time = np.random.exponential(1.0 / arrival_rate)
                await asyncio.sleep(inter_arrival_time)

        responses = await asyncio.gather(*tasks)
        end_time = time.time()

        for i, response in enumerate(responses):
            if response.status == 200:
                successful_requests += 1
                data = await response.json()
                et = data["end_time"]
                latencies.append(et - sts[i])

        total_time = end_time - start_time

    if successful_requests > 0:
        experiment_type = 'poisson' if arrival_rate != 0 else 'constant'
        average_time_per_request = total_time / successful_requests
        throughput = successful_requests / total_time

        average_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        latency_90 = sorted(latencies)[int(0.9 * len(latencies))]
        latency_95 = sorted(latencies)[int(0.95 * len(latencies))]
        latency_99 = sorted(latencies)[int(0.99 * len(latencies))]
        
        # if output file doesn't exist, create one with pandas
        try:
            df = pd.read_csv(output_file)
        except:
            df = pd.DataFrame(columns=['compute_type', 'num_workers', 'batch_size','experiment_type', 'arrival_rate', 'num_requests', 'total_time', 'successful_requests', 'average_time_per_request', 'throughput', 'average_latency', 'min_latency', 'max_latency', 'latency_90', 'latency_95', 'latency_99', 'model_id'])
        
        with open(config_file) as f:
            config = json.load(f)
            batch_size = config['server']['batch_size']
            model_id = config['server']['model_id']
        f.close()
        df = pd.concat([df, pd.DataFrame([[compute_type, num_workers, batch_size, experiment_type, arrival_rate, num_requests, total_time, successful_requests, average_time_per_request, throughput, average_latency, min_latency, max_latency, latency_90, latency_95, latency_99, model_id]], columns=df.columns)], ignore_index=True)
        df.to_csv(output_file, index=False)
    else:
        print("No successful requests")


def update_batch_size(batch_size):
    with open(config_file) as f:
        config = json.load(f)
        config['server']['batch_size'] = batch_size
    f.close()
    
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)
    f.close()

def update_model_id(model_id):
    with open(config_file) as f:
        config = json.load(f)
        config['server']['model_id'] = model_id
    f.close()
    
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)
    f.close()


if __name__ == '__main__':
    asyncio.run(measure_throughput_and_latency(url, num_requests, arrival_rate, output_file))
