import zmq 
import time 
import json  
import multiprocessing as mp 

import operator as op 
import itertools as it, functools as ft

from datetime import datetime 
from heapq import heapify, heappush, heappop 
from libraries.log import logger 
from libraries.strategies import * 

from torchvision.models import vgg16 as VGG 

class ZMQTransformer:
    def __init__(self, source, target, vgg_path, fst_router_port, snd_router_port, publisher_port, nb_workers):
        self.fst_router_port = fst_router_port
        self.snd_router_port = snd_router_port
        self.publisher_port = publisher_port
        

        self.nb_workers = nb_workers 
        self.vgg_path = vgg_path 
        self.source = source 
        self.target = target 

        self.device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')

    def get_model(self):
        model_path = path.join(self.vgg_path, 'vgg16.th')
        if path.isfile(model_path):
            return th.load(model_path)
        return None 

    def start(self):
        model = self.get_model()
        if model is not None:
            for prm in model.parameters():
                prm.requires_grad = False 
            model = model.to(self.device)

        workers_states = mp.Value('i', 0)
        workers_barrier = mp.Barrier(self.nb_workers)
        workers_condition = mp.Condition()

        responses_memory = mp.Queue()
        server_readyness = mp.Event()
        
        server_process = mp.Process(
            target=self.server, 
            args=[server_readyness, workers_states, workers_condition, responses_memory]
        )
        worker_pool = []
        for pid in range(self.nb_workers):
            worker_process = mp.Process(
                target=self.worker, 
                args=[f'{pid:03d}', server_readyness, workers_barrier, workers_states, workers_condition, responses_memory, model]
            )
            worker_pool.append(worker_process)
            worker_pool[-1].start() 

        server_process.start()
        server_process.join()

    def server(self, server_readyness, workers_states, workers_condition, responses_memory):
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
            
            server_readyness.set()  # turn on the flag(ready) and notify all workers 
            
            workers_memory = dict()
            requests_memory = []

            keep_routing = True 
            while keep_routing:
                if keep_routing:
                    if not responses_memory.empty():
                        # there is some response from worker
                        # send it to distant customer 
                        response_status, response_address, response_contents = responses_memory.get()
                        dumped_response = json.dumps({
                            'status': 'SUCCESS' if response_status == 0 else 'FAILURE',
                            'contents': response_contents
                        })
                        fst_rot_socket.send_multipart([response_address, b'', dumped_response.encode()])
                        logger.debug(f'frontend router send response to remote client with address ({response_address})')

                    fst_events = dict(fst_rot_socket_ctl.poll(100))
                    if fst_rot_socket in fst_events:
                        if fst_events[fst_rot_socket] == zmq.POLLIN:
                            # an incoming request arrived from remote customer
                            logger.debug(f'the server receives incoming request from external customer on the port {self.fst_router_port}')
                            customer_data = fst_rot_socket.recv_multipart()
                            customer_address, _, customer_message = customer_data                             
                            jsoned_customer_message = json.loads(customer_message.decode())
                            request_priority = jsoned_customer_message['priority']  
                            request_contents = jsoned_customer_message['contents']  
                            heappush(requests_memory, (request_priority, time.time(), customer_address, request_contents))
                    # check if there is incoming data from worker every 100ms
                    snd_events = dict(snd_rot_socket_ctl.poll(100))  
                    if snd_rot_socket in snd_events: 
                        if snd_events[snd_rot_socket] == zmq.POLLIN: 
                            worker_address, _, worker_message = snd_rot_socket.recv_multipart()
                            logger.debug(f'the server receives a new data ... from {worker_address.decode()}')
                            if worker_message.decode() == 'ready':
                                workers_memory[worker_address] = True 
                    
                    for worker_address, worker_status in workers_memory.items():
                        if worker_status:
                            logger.success(f'{worker_address} is available for task ...!')
                            if len(requests_memory) > 0:
                                # this line will be removed ...!
                                logger.debug('=>'.join([ p for p,_,_,_ in requests_memory ]))
                                priority, arrival_time, current_address, current_request = heappop(requests_memory)
                                arrival_time_str = datetime.fromtimestamp(arrival_time).strftime("%m_%d_%Y#%H:%M:%S")
                                snd_rot_socket.send_multipart([
                                    worker_address,
                                    b'',
                                    priority.encode(),
                                    b'',
                                    arrival_time_str.encode(),  
                                    b'',
                                    current_address, 
                                    b'',
                                    json.dumps(current_request).encode(),
                                ])
                                workers_memory[worker_address] = False # worker is busy :(

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

    def worker(self, pid, server_readyness, workers_barrier, workers_states, workers_condition, responses_memory, model):
        try:
            print(model)
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

            mapper = get_mapper()

            my_status = True 
            keep_processing = True 
            while keep_processing:
                if my_status:
                    dea_socket.send_multipart([b'', b'ready'])
                    my_status = False 
                dealer_events = dict(dea_socket_ctl.poll(100))
                if dea_socket in dealer_events:
                    if dealer_events[dea_socket] == zmq.POLLIN:
                        _, priority, _, arrival_time, _, current_address, _, current_request = dea_socket.recv_multipart()
                        try:
                            loaded_request = json.loads(current_request.decode())
                            path2image = loaded_request['path2image']
                            transformation2apply = loaded_request['transformation']
                            logger.debug(f'worker ({pid}) apply {transformation2apply} on {path2image}')
                
                            _, image_filename = path.split(path2image)
                            image_local_path = path.join(self.source, image_filename)

                            if path.isfile(image_local_path):
                                cv_image = read_image(image_local_path)
                                th_image = cv2th(cv_image)
                                mapped_image = mapper[transformation2apply](th_image)
                                path2imagedump = path.join(self.target, f'{priority.decode()}_{arrival_time.decode()}_{image_filename}')
                                prepared_image = prepare_image(mapped_image).to(self.device)
                                extracted_features = model(prepared_image[None, ...]).cpu().numpy()
                                logger.success(f'({pid}) feature extraction {extracted_features.shape}')
                                print(extracted_features)
                                save_image(th2cv(mapped_image), path2imagedump)
                                logger.success(f'worker ({pid}) saves the result on {path2imagedump}')
                                responses_memory.put((0, current_address, path2imagedump))
                            else:
                                logger.error(f'{path2image} is not a valid path ...!')
                                responses_memory.put((1, current_address, 'no image was saved ...!'))

                        except Exception as e:
                            logger.error(f'an exception occurs during processing by the worker ({pid}) ...!')
                            logger.warning(e)
                         
                        my_status = True  # worker is ready again to receive a new task 
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
