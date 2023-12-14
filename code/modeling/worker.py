"""Apply sinthetisation with SMOTE
This script will apply SMOTE technique in the single out cases.
"""
# %%
#!/usr/bin/env python
import warnings
import functools
import threading
import argparse
from apply_models import modeling
import pika

warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
#%%
parser = argparse.ArgumentParser(description='Master Example')
parser.add_argument('--type', type=str, help='Strategy type', default="ppt")
parser.add_argument('--opt', type=str, help='Optimization strategy', default="BO")
parser.add_argument('--input_folder', type=str, help='Input folder', default="./input")
parser.add_argument('--output_folder', type=str, help='Output folder', default="./output")
args = parser.parse_args()

def ack_message(ch, delivery_tag, work_sucess):
    """Acknowledge message

    Args:
        ch (_type_): channel of the queue
        delivery_tag (_type_): _description_
        work_sucess (_type_): _description_
    """
    if ch.is_open:
        if work_sucess:
            print("[x] Done")
            ch.basic_ack(delivery_tag)
        else:
            ch.basic_reject(delivery_tag, requeue=False)
            print("[x] Rejected")
    else:
        # Channel is already closed, so we can't ACK this message;
        # log and/or do something that makes sense for your app in this case.
        pass


def modeling_(file):
    modeling(file, args)

# %%
def do_work(conn, ch, delivery_tag, body):
    msg = body.decode('utf-8')
    work_sucess = modeling_(msg)
    cb = functools.partial(ack_message, ch, delivery_tag, work_sucess)
    conn.add_callback_threadsafe(cb)


def on_message(ch, method_frame, _header_frame, body, args):
    (conn, thrds) = args
    delivery_tag = method_frame.delivery_tag
    t = threading.Thread(target=do_work, args=(conn, ch, delivery_tag, body))
    t.start()
    thrds.append(t)


#credentials = pika.PlainCredentials('guest', 'guest')
connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost', heartbeat=5))
channel = connection.channel()
channel.queue_declare(queue='task_queue', durable=True, arguments={"dead-letter-exchange":"dlx"})
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
# python3 code/modeling/task.py --input_folder "data/PrivateSMOTEk2"
# python3 code/modeling/task.py --input_folder "data/original"
# python3 code/modeling/task.py --input_folder "data/synthcityk2"
# python3 code/modeling/worker.py --type "ppt" --opt "BO" --input_folder "data/PPT_transformed/PPT_train" --output_folder "output/modelingBO/PPT_ARX_bo100"
# python3 code/modeling/worker.py --type "ppt" --opt "RS" --input_folder "data/PPT_transformed/PPT_train" --output_folder "output/modelingRS/PPT_ARX"
# python3 code/modeling/worker.py --type "gans" --opt "BO" --input_folder "data/deep_learningk2" --output_folder "output/modelingBO/deep_learningk2"
# python3 code/modeling/worker.py --type "gans" --opt "HB" --input_folder "data/deep_learningk2" --output_folder "output/modelingHB/deep_learningk2"
# python3 code/modeling/worker.py --type "gans" --opt "SH" --input_folder "data/deep_learningk2" --output_folder "output/modelingSH/deep_learningk2"
# python3 code/modeling/worker.py --type "gans" --opt "GS" --input_folder "data/deep_learningk2" --output_folder "output/modelingGS/deep_learningk2"
# python3 code/modeling/worker.py --type "gans" --opt "RS" --input_folder "data/deep_learningk2" --output_folder "output/modelingRS/deep_learningk2"
# python3 code/modeling/worker.py --type "PrivateSMOTE" --opt "HB" --input_folder "data/PrivateSMOTEk2" --output_folder "output/modelingHB/PrivateSMOTEk2"
# python3 code/modeling/worker.py --type "synthcity" --opt "BO" --input_folder "data/synthcityk2" --output_folder "output/modelingBO/synthcityk2"
# python3 code/modeling/worker.py --type "original" --opt "HB" --input_folder "data/original" --output_folder "output/modelingHB/original"