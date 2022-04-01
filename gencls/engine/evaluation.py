from multiprocessing import get_logger
import time
import torch
import tqdm 


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
    res_metric_eval = engine.metric_info.avg

    engine.logger.info(f"[Eval][Epoch {epoch_id}/{engine.config['Common']['epochs']}]: {eval_msg}")
    engine.writer.add_scalar('val_loss', engine.loss_info.avg, epoch_id)
    engine.writer.add_scalar('acc_eval', res_metric_eval, epoch_id)
    engine.loss_info.reset()
    engine.metric_info.reset()
    # for key in engine.time_info.keys():
    #     engine.time_info[key].reset()

    return res_metric_eval

def test(engine):
    tic = time.time()
    outputs_all = None 
    targets_all = None
    with torch.no_grad():
        for idx, batch_data in tqdm.tqdm(enumerate(engine.val_dataloader)):
            
            batch_data = engine.transfer_batch_to_device(batch_data)
            targets = batch_data['label'].long()
            outputs = engine.model(batch_data['img'])

            outputs_all = torch.cat((outputs_all, outputs)) if outputs_all is not None else outputs

            targets_all = torch.cat((targets_all, targets)) if targets_all is not None else targets
            
    assert outputs_all.size(0) == targets_all.size(0)
    metric_result = engine.eval_metric_func(outputs_all, targets_all)
    return metric_result


        

