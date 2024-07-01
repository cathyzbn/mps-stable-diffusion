
import aiohttp
import asyncio
import time
import numpy as np
import pandas as pd
import json
import os


np.random.seed(0)


config_file = 'config.json'
with open(config_file) as f:
    config = json.load(f)
    
    num_requests = config['request_generator']['num_requests']

    compute_type = config['request_generator']['compute_type']
    output_file = config['request_generator']['output_file']
    prompt = config['request_generator']['prompt']
    url = config['request_generator']['url']
    batch_size = config['server']['batch_size']
f.close()


input = "%20".join(prompt.split(" "))

# if arrival_rate != 0:
#     inter_arrival_times = np.random.exponential(1.0/arrival_rate, num_requests)

async def send_get_request(session, url, input):
    start_time = time.time()
    request = session.get(f"{url}?prompt={input}")
    task = asyncio.create_task(request)
    
    #await the request to ensure it's sent
    await asyncio.sleep(0)
    return task, start_time


async def measure_throughput_and_latency(url, num_requests, output_file, arrival_rate):
    total_time = 0
    successful_requests = 0

    async with aiohttp.ClientSession() as session:
        warmer_tasks = []
        for i in range(16):
            task, st = await send_get_request(session, url, input)
            warmer_tasks.append(task)
        await asyncio.gather(*warmer_tasks)
            

        tasks = []
        sts = []
        latencies = []
        queue_time = []
        execution_time = []

        start_time = time.time()

        with open(config_file) as f:
            config = json.load(f)
            arrival_rate = config['request_generator']['arrival_rate']
        
        arrival_times = [arrival_rate for i in range(num_requests)] if arrival_rate != 0 else None      
        
        for i in range(num_requests):
            task, st = await send_get_request(session, url, input)

            tasks.append(task)
            sts.append(st)

            if arrival_times is not None:
                await asyncio.sleep(arrival_times[i])

        responses = await asyncio.gather(*tasks)
        end_time = time.time()

        for i, response in enumerate(responses):
            if response.status == 200:
                successful_requests += 1
                data = await response.json()
                et = data["end_time"]
                server_start_time = data["server_start_time"]
                latencies.append(et - sts[i])
                queue_time.append(server_start_time - sts[i])
                execution_time.append(et - server_start_time)



        total_time = end_time - start_time

    if successful_requests > 0:
        experiment_type = 'poisson' if arrival_times is not None else 'constant'
        average_time_per_request = total_time / successful_requests
        throughput = successful_requests / total_time

        average_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        latency_90 = sorted(latencies)[int(0.9 * len(latencies))]
        latency_95 = sorted(latencies)[int(0.95 * len(latencies))]
        latency_99 = sorted(latencies)[int(0.99 * len(latencies))]
        avg_queue_time = sum(queue_time) / len(queue_time)
        avg_execution_time = sum(execution_time) / len(execution_time)
        print("Latencies", latencies)
        print("Queue Times", queue_time)
        print("Execution Times", execution_time)


        with open(config_file) as f:
            config = json.load(f)
            batch_size = config['server']['batch_size']
            model_id = config['server']['model_id']
            num_workers = config['server']['num_workers']

        # # display full log of each request
        # full_csv_path = f"full_logs/w{num_workers}_bs{batch_size}_{compute_type}_{arrival_rate}.csv"
        # df = pd.DataFrame(columns=['request_id', 'start_time', 'end_time', 'queue_time', 'execution_time', 'latency'])
        # for i in range(len(latencies)):
        #     df = pd.concat([df, pd.DataFrame([[i, sts[i], sts[i] + latencies[i], queue_time[i], execution_time[i], latencies[i]]], columns=df.columns)], ignore_index=True)
        # df.to_csv(full_csv_path, index=False)

        try:
            df = pd.read_csv(output_file)
        except:
            df = pd.DataFrame(columns=['compute_type', 'num_workers', 'batch_size','experiment_type', 'arrival_rate', 'num_requests', 'total_time', 'successful_requests', 'average_time_per_request', 'throughput', 'average_latency', 'min_latency', 'max_latency', 'latency_90', 'latency_95', 'latency_99', 'model_id', 'avg_queue_time', 'avg_execution_time'])
        
        f.close()
        df = pd.concat([df, pd.DataFrame([[compute_type, num_workers, batch_size, experiment_type, arrival_rate, num_requests, total_time, successful_requests, average_time_per_request, throughput, average_latency, min_latency, max_latency, latency_90, latency_95, latency_99, model_id, avg_queue_time, avg_execution_time]], columns=df.columns)], ignore_index=True)
        df.to_csv(output_file, index=False)
    else:
        print("No successful requests")


def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")


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

def update_num_workers(num_workers):
    with open(config_file) as f:
        config = json.load(f)
        config['server']['num_workers'] = num_workers
    f.close()
    
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)
    f.close()

def update_arrival_rate(arrival_rate):
    with open(config_file) as f:
        config = json.load(f)
        config['request_generator']['arrival_rate'] = arrival_rate
    f.close()
    
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)
    f.close()