"""Task manager
This script will add the tasks in the queue
"""
#!/usr/bin/env python
import argparse
import os
import re
import pika
import pandas as pd
import ast

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
        routing_key='task_queue',
        body=file_name,
        properties=pika.BasicProperties(
            delivery_mode=2,  # make message persistent
        ))


credentials = pika.PlainCredentials('guest', 'guest')
connection = pika.BlockingConnection(
    pika.ConnectionParameters(host='localhost', credentials=credentials, heartbeat=5))
channel = connection.channel()

channel.exchange_declare(exchange='dlx', exchange_type='direct')

channel.queue_declare(queue='task_queue', durable=True, arguments={"dead-letter-exchange": "dlx"})
dl_queue = channel.queue_declare(queue='dl')

channel.queue_bind(exchange='dlx', routing_key='task_queue', queue=dl_queue.method.queue)

file_list = [f for f in os.listdir(args.input_folder) if '.csv' in f]
for file in file_list:
    file = file.split('.')[0]
    f = list(map(int, re.findall(r'\d+', file.split('_')[0])))[0]
    if f not in [0,1,3,13,23,28,34,36,40,48,54,66,87, 100,43]:
        # filter QIs set in PPT w/ ARX
        if (len(file.split('_')) < 4) and len(file) > 2:
            keys_nr = list(map(int, re.findall(r'\d+', file.split('_')[2])))[0]
            if keys_nr < 3:
                print(file)
                put_file_queue(channel, file)

        else:
            print(file)
            put_file_queue(channel, file)

connection.close()
