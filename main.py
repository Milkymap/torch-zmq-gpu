import os 
import click 
import subprocess 

from libraries.log import logger 
from processing.process import ZMQTransformer

def start():
    source = os.getenv("SOURCE")
    nb_workers = int(os.getenv("NB_WORKERS"))
    publisher_port = int(os.getenv("PUBLISHER_PORT"))
    back_router_port = int(os.getenv("BACK_ROUTER_PORT"))
    front_router_port = int(os.getenv("FRONT_ROUTER_PORT"))
    logger.success('all environments variables ware loaded ...!')
    zmq_obj = ZMQTransformer(source, front_router_port, back_router_port, publisher_port, nb_workers)
    zmq_obj.start()

if __name__ == '__main__':
    start()

