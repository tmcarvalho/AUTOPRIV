"""Task manager
This script will add the tasks in the queue
"""
#!/usr/bin/env python
import argparse
import os
import pika

parser = argparse.ArgumentParser(description='Master Example')
parser.add_argument('--input_folder', type=str, help='Input folder', default="./input")
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

epochs=[100, 200]
batch_size=[50, 100]

files = files = next(os.walk(args.input_folder))[2]
# SDV
# for file in files:
#     f = int(file.split('.')[0])
#     if f not in [0,1,3,13,23,28,34,36,40,48,54,66,87, 100,43]:
#         print(file)
#         for technique in ['CTGAN', 'CopulaGAN', 'TVAE']:
#             for idx in range(3):
#                 for ep in epochs:
#                     for bs in batch_size:
#                         print(f'ds{file.split(".")[0]}_{technique}_QI{idx}_ep{ep}_bs{bs}')
#                         put_file_queue(channel, f'ds{file.split(".")[0]}_{technique}_QI{idx}_ep{ep}_bs{bs}')

# SYNTHCITY
n_iter=[100, 200]
batch_size_=[50, 100]
epsilon=[0.1, 0.25, 0.5, 0.75, 1.0]
learning_iter=[100, 250, 500, 750, 1000, 1500, 2000]

for file in files:
    f = int(file.split('.')[0])
    if f not in [0,1,3,13,23,28,34,36,40,48,54,66,87, 100,43]:
        print(file)
        for technique in ['dpgan', 'pategan']:
            for idx in range(3):
                for epo in n_iter:
                    for bs in batch_size_:
                        for epi in epsilon:
                            print(f'ds{file.split(".")[0]}_{technique}_QI{idx}_epo{epo}_bs{bs}_epi{epi}')
                            put_file_queue(channel, f'ds{file.split(".")[0]}_{technique}_QI{idx}_epo{epo}_bs{bs}_epi{epi}')

        for idx in range(3):
            for epi in epsilon:
                print(f'ds{file.split(".")[0]}_privbayes_QI{idx}_epi{epi}')
                put_file_queue(channel, f'ds{file.split(".")[0]}_privbayes_QI{idx}_epi{epi}')

        # for technique in ['tvae', 'ctgan']:
        #     for idx in range(3):
        #         for epo in n_iter:
        #             for bs in batch_size_:
        #                     print(f'ds{file.split(".")[0]}_{technique}_QI{idx}_epo{epo}_bs{bs}')
        #                     put_file_queue(channel, f'ds{file.split(".")[0]}_{technique}_QI{idx}_epo{epo}_bs{bs}')

        # for idx in range(3):
        #     for li in learning_iter:
        #         print(f'ds{file.split(".")[0]}_bayesiannetwork_QI{idx}_li{li}')
        #         put_file_queue(channel, f'ds{file.split(".")[0]}_bayesian_network_QI{idx}_li{li}')


connection.close()