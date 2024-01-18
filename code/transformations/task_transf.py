"""Task manager
This script will add the tasks in the queue
"""
#!/usr/bin/env python
import argparse
import os
import pika

parser = argparse.ArgumentParser(description='Master Example')
parser.add_argument('--input_folder', type=str, help='Input folder', default="./input")
parser.add_argument('--type', type=str, help='Technique', default="None")
args = parser.parse_args()


def put_file_queue(ch, file_name):
    """Add files to the queue

    Args:
        ch (_type_): channel of the queue
        file_name (string): name of file
    """
    ch.basic_publish(
        exchange='',
        routing_key='task_queue_transf',
        body=file_name,
        properties=pika.BasicProperties(
            delivery_mode=2,  # make message persistent
        ))


credentials = pika.PlainCredentials('guest', 'guest')
connection = pika.BlockingConnection(
    pika.ConnectionParameters(host='localhost', credentials=credentials, heartbeat=5))
channel = connection.channel()

channel.exchange_declare(exchange='dlx', exchange_type='direct')

channel.queue_declare(queue='task_queue_transf', durable=True, arguments={"dead-letter-exchange": "dlx"})
dl_queue = channel.queue_declare(queue='dl')

channel.queue_bind(exchange='dlx', routing_key='task_queue_transf', queue=dl_queue.method.queue)

files = next(os.walk(args.input_folder))[2]

if args.type == 'PrivateSMOTE':
    knn = [1,3,5]
    per = [1,2,3]
    epislon = [0.1, 0.5, 1.0, 5.0, 10.0]

    for file in files:
        f = int(file.split('.csv')[0])
        if f not in [0,1,3,13,23,28,34,36,40,48,54,66,87, 100,43]:
        # if f in [37]:
            print(file)
            for idx in range(5):
                for k in knn:
                    for p in per:
                        for ep in epislon:
                            print(f'ds{file.split(".")[0]}_{ep}-privateSMOTE_QI{idx}_knn{k}_per{p}')
                            put_file_queue(channel, f'ds{file.split(".")[0]}_{ep}-privateSMOTE_QI{idx}_knn{k}_per{p}')   

if args.type == 'SDV':
    epochs=[100, 200]
    batch_size=[50, 100]

    for file in files:
        f = int(file.split('.csv')[0])
        if f not in [0,1,3,13,23,28,34,36,40,48,54,66,87, 100,43]:
            print(file)
            for technique in ['CTGAN', 'CopulaGAN', 'TVAE']:
                for idx in range(5):
                    for ep in epochs:
                        for bs in batch_size:
                            print(f'ds{file.split(".")[0]}_{technique}_QI{idx}_ep{ep}_bs{bs}')
                            put_file_queue(channel, f'ds{file.split(".")[0]}_{technique}_QI{idx}_ep{ep}_bs{bs}')

if args.type == 'Synthcity':
    n_iter=[100, 200]
    batch_size_=[50, 100]
    epsilon=[0.1, 0.5, 1.0, 5.0]

    for file in files:
        f = int(file.split('.csv')[0])
        if f not in [0,1,3,13,23,28,34,36,40,48,54,66,87, 100,43]:
            print(file)
            for technique in ['dpgan', 'pategan']:
                for idx in range(5):
                    for epo in n_iter:
                        for bs in batch_size_:
                            for epi in epsilon:
                                print(f'ds{file.split(".")[0]}_{technique}_QI{idx}_epo{epo}_bs{bs}_epi{epi}')
                                put_file_queue(channel, f'ds{file.split(".")[0]}_{technique}_QI{idx}_epo{epo}_bs{bs}_epi{epi}')

connection.close()