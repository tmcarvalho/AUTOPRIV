"""Task manager
This script will add the tasks in the queue
"""
#!/usr/bin/env python
import argparse
import os
import pika
import re
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
        routing_key='task_queue_anon',
        body=file_name,
        properties=pika.BasicProperties(
            delivery_mode=2,  # make message persistent
        ))


credentials = pika.PlainCredentials('guest', 'guest')
connection = pika.BlockingConnection(
    pika.ConnectionParameters(host='localhost', credentials=credentials, heartbeat=5))
channel = connection.channel()

channel.exchange_declare(exchange='dlx', exchange_type='direct')

channel.queue_declare(queue='task_queue_anon', durable=True, arguments={"dead-letter-exchange": "dlx"})
dl_queue = channel.queue_declare(queue='dl')

channel.queue_bind(exchange='dlx', routing_key='task_queue_anon', queue=dl_queue.method.queue)

for file in os.listdir(args.input_folder):
    f = list(map(int, re.findall(r'\d+', file.split('_')[0])))[0]
    if f not in [0,1,3,13,23,28,34,36,40,48,54,66,87, 100,43]:
        if len(file.split('_')) < 4:
            list_key_vars = pd.read_csv('list_key_vars.csv')
            set_key_vars = ast.literal_eval(
                list_key_vars.loc[list_key_vars['ds']==f[0], 'set_key_vars'].values[0])

            keys_nr = list(map(int, re.findall(r'\d+', file.split('_')[2])))[0]

            if keys_nr < 3:
                print(file)
                put_file_queue(channel, file)

        else:
            print(file)
            put_file_queue(channel, file)

connection.close()