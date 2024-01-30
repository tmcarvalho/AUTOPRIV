"""Apply sinthetisation with SMOTE
This script will apply SMOTE technique in the single out cases.
"""
# %%
#!/usr/bin/env python
#!/usr/bin/env python
import warnings
import functools
import threading
import argparse
import pika
from apply_models import modeling

warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser(description='Master Example')
parser.add_argument('--type', type=str, help='Strategy type', default="ppt")
parser.add_argument('--opt', type=str, help='Optimization strategy', default="BO")
parser.add_argument('--input_folder', type=str, help='Input folder', default="./input")
parser.add_argument('--output_folder', type=str, help='Output folder', default="./output")
args = parser.parse_args()

def ack_message(ch, delivery_tag, work_success):
    """Acknowledge message"""
    if ch.is_open:
        if work_success:
            print("[x] Done")
            ch.basic_ack(delivery_tag)
        else:
            ch.basic_reject(delivery_tag, requeue=False)
            print("[x] Rejected")
    else:
        pass

def modeling_(file):
    success = modeling(file, args)
    return success

def do_work(conn, ch, delivery_tag, body):
    msg = body.decode('utf-8')
    work_success = modeling_(msg)
    cb = functools.partial(ack_message, ch, delivery_tag, work_success)
    conn.add_callback_threadsafe(cb)

def on_message(ch, method_frame, _header_frame, body, args):
    (conn, thrds) = args
    delivery_tag = method_frame.delivery_tag
    t = threading.Thread(target=do_work, args=(conn, ch, delivery_tag, body))
    t.start()
    thrds.append(t)

connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost', heartbeat=5))
channel = connection.channel()
channel.queue_declare(queue='task_queue', durable=True, arguments={"dead-letter-exchange": "dlx"})
print(' [*] Waiting for messages. To exit press CTRL+C')

channel.basic_qos(prefetch_count=1)

threads = []
on_message_callback = functools.partial(on_message, args=(connection, threads))
channel.basic_consume('task_queue', on_message_callback)

try:
    channel.start_consuming()
except KeyboardInterrupt:
    channel.stop_consuming()

# Wait for all to complete
for thread in threads:
    thread.join()

connection.close()



# find . -name ".DS_Store" -delete
# python3 code/modeling/task.py --input_folder "data/PPT_transformed/PPT_train"
# python3 code/modeling/task.py --input_folder "data/PrivateSMOTE"
# python3 code/modeling/task.py --input_folder "data/original"
# python3 code/modeling/task.py --input_folder "data/deep_learning"
# python3 code/modeling/worker.py --type "ppt" --opt "BO" --input_folder "data/PPT_transformed/PPT_train" --output_folder "output/modelingBO/PPT_ARX"
# python3 code/modeling/worker.py --type "ppt" --opt "RS" --input_folder "data/PPT_transformed/PPT_train" --output_folder "output/modelingRS/PPT_ARX"
# python3 code/modeling/worker.py --type "gans" --opt "BO" --input_folder "data/deep_learning" --output_folder "output/modelingBO/deep_learning"
# python3 code/modeling/worker.py --type "gans" --opt "HB" --input_folder "data/deep_learning" --output_folder "output/modelingHB/deep_learning"
# python3 code/modeling/worker.py --type "gans" --opt "SH" --input_folder "data/deep_learning" --output_folder "output/modelingSH/deep_learning"
# python3 code/modeling/worker.py --type "gans" --opt "GS" --input_folder "data/deep_learning" --output_folder "output/modelingGS/deep_learning"
# python3 code/modeling/worker.py --type "gans" --opt "RS" --input_folder "data/deep_learning" --output_folder "output/modelingRS/deep_learning"
# python3 code/modeling/worker.py --type "PrivateSMOTE" --opt "RS" --input_folder "data/PrivateSMOTE" --output_folder "output/modelingRS/PrivateSMOTE"
# python3 code/modeling/worker.py --type "synthcity" --opt "RS" --input_folder "data/synthcityk2" --output_folder "output/modelingRS/synthcityk2"
# python3 code/modeling/worker.py --type "original" --opt "HB" --input_folder "data/original" --output_folder "output/modelingHB/original"