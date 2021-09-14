import os 
import click 
from libraries.log import logger 
from processing.process import ZMQTransformer
import multiprocessing as mp 

# git key ...! 
# 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

def start():
    vgg_path = os.getenv('VGG_PATH')
    source = os.getenv("SOURCE")
    target = os.getenv('TARGET')

    nb_workers = int(os.getenv("NB_WORKERS"))
    publisher_port = int(os.getenv("PUBLISHER_PORT"))
    back_router_port = int(os.getenv("BACK_ROUTER_PORT"))
    front_router_port = int(os.getenv("FRONT_ROUTER_PORT"))
    
    logger.success('all environments variables ware loaded ...!')
    zmq_obj = ZMQTransformer(source, target, vgg_path, front_router_port, back_router_port, publisher_port, nb_workers)
    zmq_obj.start()

if __name__ == '__main__':
    mp.set_start_method('spawn')
    start()


