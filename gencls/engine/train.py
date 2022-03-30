from multiprocessing import get_logger
import time

from torch.cuda import amp 


def train_one_epoch(engine, epoch_id, print_batch_step):
    tic = time.time()
    for iter_id, batch_data in enumerate(engine.train_dataloader):
        # print('{}MB allocated'.format(torch.cuda.memory_allocated()/1024**2))
        batch_data = engine.transfer_batch_to_device(batch_data)
        data_time = time.time() - tic
        targets = batch_data['label'].long()

        tic = time.time()
        if engine.grad_scaler is None: 
            outputs = engine.model(batch_data['img'])
            forward_time = time.time() - tic

            tic = time.time()
            loss = engine.criterion(outputs, targets)
            engine.optimizer.zero_grad()
            loss.backward()
            engine.optimizer.step()
            
        else:
            with amp.autocast():
                outputs = engine.model(batch_data['img'])
                forward_time = time.time() - tic

                tic = time.time()
                loss = engine.criterion(outputs, targets)
            engine.grad_scaler.scale(loss).backward()
            engine.grad_scaler.step(engine.optimizer)
            engine.grad_scaler.update()

        if engine.scheduler is not None:
            engine.scheduler.step()
        backward_time = time.time() - tic
        engine.time_info['data_time'].update(data_time)
        engine.time_info['forward_time'].update(forward_time)
        engine.time_info['backward_time'].update(backward_time)
        engine.loss_info.update(loss.item(), outputs.size(0))

        if iter_id != 0 and iter_id % print_batch_step == 0:
            time_msg = "s, ".join([
                "{}: {:.4f}".format(key, engine.time_info[key].avg)
                for key in engine.time_info.keys()
            ])

            loss_msg = f"loss: {engine.loss_info.avg}"
            engine.logger.info(f"[Train][Epoch {epoch_id}/{engine.config['Common']['epochs']}][Iter: {iter_id}/{len(engine.train_dataloader)}]: {time_msg}, {loss_msg}")

            for key in engine.time_info.keys():
                engine.time_info[key].reset()



        

