from multiprocessing import get_logger
import time
import torch 


def evaluate(engine, epoch_id):
    tic = time.time()
    with torch.no_grad():
        for idx, batch_data in enumerate(engine.val_dataloader):
            
            batch_data = engine.transfer_batch_to_device(batch_data)
            targets = batch_data['label'].long()
            outputs = engine.model(batch_data['img'])

            loss = engine.criterion(outputs, targets)
            acc = engine.eval_metric_func(outputs, targets)
            engine.metric_info.update(acc, outputs.size(0))
            engine.loss_info.update(loss.item(), outputs.size(0))
    eval_time = time.time() - tic
    eval_msg = "Time_eval: {:.2f} - Loss_eval: {:.4f} - Acc_eval: {:.2f}".format(eval_time, engine.loss_info.avg, engine.metric_info.avg)


    engine.logger.info(f"[Eval][Epoch {epoch_id}/{engine.config['Common']['epochs']}]: {eval_msg}")

    # for key in engine.time_info.keys():
    #     engine.time_info[key].reset()

    return engine.metric_info.avg


        

