import zmq 
import time 
import click 
import multiprocessing as mp 

import operator as op 
import itertools as it, functools as ft

from libraries.log import logger 
from libraries.strategies import * 


class ZMQTransformer:
    def __init__(self, source, fst_router_port, snd_router_port, publisher_port, nb_workers):
        self.fst_router_port = fst_router_port
        self.snd_router_port = snd_router_port
        self.publisher_port = publisher_port

        self.nb_workers = nb_workers 
        self.source = source 
        self.orders = ['vflip', 'hflip', 'ccrop', 'cgray', 'trota']

    def start(self):
        workers_states = mp.Value('i', 0)
        workers_barrier = mp.Barrier(self.nb_workers)
        workers_condition = mp.Condition()

        server_readyness = mp.Event()
        
        server_process = mp.Process(target=self.server, args=[server_readyness, workers_states, workers_condition])
        worker_pool = []
        for pid in range(self.nb_workers):
            worker_process = mp.Process(
                target=self.worker, 
                args=[f'{pid:03d}', server_readyness, workers_barrier, workers_states, workers_condition]
            )
            worker_pool.append(worker_process)
            worker_pool[-1].start() 

        server_process.start()
        
    def server(self, server_readyness, workers_states, workers_condition):
        try:   
            ctx = zmq.Context()

            fst_rot_socket = ctx.socket(zmq.ROUTER)
            fst_rot_socket_ctl = zmq.Poller()

            snd_rot_socket = ctx.socket(zmq.ROUTER)
            snd_rot_socket_ctl = zmq.Poller()
            
            fst_rot_socket.bind(f'tcp://*:{self.fst_router_port}')
            snd_rot_socket.bind(f'tcp://*:{self.snd_router_port}')
            
            fst_rot_socket_ctl.register(fst_rot_socket, zmq.POLLIN)
            snd_rot_socket_ctl.register(snd_rot_socket, zmq.POLLIN)

            logger.success('server router initialization complete ...!')
            
            pub_socket = ctx.socket(zmq.PUB)
            pub_socket.bind(f'tcp://*:{self.publisher_port}')
            logger.success('server publisher initialization complete ...!')
            
            server_readyness.set()  # turn on the flag and notify all workers that we are ready 
            
            filepaths = pull_files(self.source, '*.jpg')
            nb_images = len(filepaths)
            file_cursor = nb_images 
            logger.debug(f'internal database has {nb_images} images')

            workers_memory = dict()
            requests_memory = mp.Queue()
            keep_routing = True 
            while keep_routing:
                if keep_routing:
                    fst_events = dict(fst_rot_socket_ctl.poll(100))
                    if fst_rot_socket in fst_events:
                        if fst_events[fst_rot_socket] == zmq.POLLIN:
                            # an incoming request arrived from remote customer
                            logger.debug(f'the server receive incoming request from external customer on the port {self.fst_router_port}')
                            customer_data = fst_rot_socket.recv_multipart()
                            customer_address, _, customer_message = customer_data
                            requests_memory.put((customer_address, customer_message))
                            logger.debug(customer_message.decode())
                            if customer_message.decode() == "QUIT":
                                keep_routing = False
                    # check if there is incoming data from worker every 100ms
                    snd_events = dict(snd_rot_socket_ctl.poll(100))  
                    if snd_rot_socket in snd_events:  # incomplete logic if code : 
                        if snd_events[snd_rot_socket] == zmq.POLLIN: 
                            worker_address, _, worker_message = snd_rot_socket.recv_multipart()
                            logger.debug(f'the server receives a new data ... from {worker_address.decode()}')
                            if worker_message.decode() == 'ready':
                                workers_memory[worker_address] = True 
                    
                    for worker_address, worker_status in workers_memory.items():
                        if worker_status:
                            logger.success(f'{worker_address} is available for task ...!')
                            if not requests_memory.empty():
                                current_address, current_request = requests_memory.get()
                                target_order = np.random.choice(self.orders)
                                snd_rot_socket.send_multipart([
                                    worker_address, 
                                    b'',
                                    current_address, 
                                    b'',
                                    current_request,
                                    b'',
                                    target_order.encode()
                                ])
                                workers_memory[worker_address] = False # busy worker :(

                else:
                    logger.success('the server finish to send images ...!')
            # end routing loop 

            pub_socket.send_multipart([b'TERMINATE', b''])

        except KeyboardInterrupt as e:
            logger.warning('keyboard interrupt') 
        except zmq.ZMQError as e:
            logger.error(e)
        except Exception as e:
            logger.error(e)
        finally:
            logger.debug('server is waiting for workers to disconnect ...!')
            workers_condition.acquire() # acquire the underlying lock 
            workers_condition.wait_for(lambda: workers_states.value == self.nb_workers) 
            logger.debug('server will close ...!')

            fst_rot_socket_ctl.unregister(fst_rot_socket)
            snd_rot_socket_ctl.unregister(snd_rot_socket)

            fst_rot_socket.close()
            snd_rot_socket.close()
            pub_socket.close()

            ctx.term()

            logger.success('server close successfully ...!')            

    def worker(self, pid, server_readyness, workers_barrier, workers_states, workers_condition):
        try:
            logger.debug(f'the worker ({pid}) is waiting for the server to be ready')
            server_readyness.wait()

            ctx = zmq.Context()
            dea_socket = ctx.socket(zmq.DEALER)
            dea_socket_ctl = zmq.Poller()
            
            dea_socket.setsockopt_string(zmq.IDENTITY, pid)
            dea_socket.connect(f'tcp://localhost:{self.snd_router_port}')
            dea_socket_ctl.register(dea_socket, zmq.POLLIN)

            logger.success(f'the worker ({pid}) is connected to the router ...!')
            
            sub_socket = ctx.socket(zmq.SUB)
            sub_socket.connect(f'tcp://localhost:{self.publisher_port}')
            sub_socket_ctl = zmq.Poller()
            sub_socket_ctl.register(sub_socket, zmq.POLLIN)

            sub_socket.setsockopt_string(zmq.SUBSCRIBE, 'TERMINATE')

            logger.success(f'the worker ({pid}) is connected to the publisher ...!')

            workers_barrier.wait()  # wait for others workers to be ready 

            my_status = True 
            keep_processing = True 
            while keep_processing:
                if my_status:
                    dea_socket.send_multipart([b'', b'ready'])
                    my_status = False 
                dealer_events = dict(dea_socket_ctl.poll(100))
                if dea_socket in dealer_events:
                    if dealer_events[dea_socket] == zmq.POLLIN:
                        _, current_address, _, current_request, _, order = dea_socket.recv_multipart()
                        logger.debug(f'worker ({pid}) apply {order.decode()} on {current_request.decode()}')
                        time.sleep(1)  # similate a work 
                        my_status = True  # i am ready again to receive new task 
                subscriber_events = dict(sub_socket_ctl.poll(100))
                
                if sub_socket in subscriber_events:
                    if subscriber_events[sub_socket] == zmq.POLLIN:
                        topic, _ = sub_socket.recv_multipart()
                        if topic.decode() == 'TERMINATE':
                            logger.debug(f'the worker ({pid}) got a KILL signal ...! ')
                            keep_processing = False 

            # end processing loop 
        except KeyboardInterrupt as e:
            logger.warning('keyboard interrupt') 
        except zmq.ZMQError as e:
            logger.error(e)
        except Exception as e:
            logger.error(e)
        finally:
            logger.debug(f'the worker ({pid}) will close and free its ressources ...!')
            dea_socket_ctl.unregister(dea_socket)
            dea_socket.close()
            
            sub_socket_ctl.unregister(sub_socket)
            sub_socket.close()

            ctx.term()

            workers_condition.acquire()
            with workers_states.get_lock():
                workers_states.value = workers_states.value + 1 
                logger.success(f'the worker ({pid}) close succesfully ...!')

            workers_condition.notify_all()
            workers_condition.release()
            
if __name__ == '__main__':
    logger.debug('image processing app')
