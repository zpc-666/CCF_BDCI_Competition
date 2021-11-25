# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import os.path as osp
from visualdl import LogWriter
import paddle
import os
import numpy as np
import pandas as pd
import paddle.distributed as dist
import paddle.distributed.fleet as fleet
from ..loader.builder import build_dataloader, build_dataset
from ..modeling.builder import build_model
from ..solver import build_lr, build_optimizer
from ..utils import do_preciseBN
from paddlevideo.utils import get_logger
from paddlevideo.utils import (build_record, log_batch, log_epoch, save, load,
                               mkdir)


def train_model(cfg,
                weights=None,
                parallel=True,
                validate=True,
                amp=False,
                use_fleet=False):
    """Train model entry

    Args:
    	cfg (dict): configuration.
        weights (str): weights path for finetuning.
    	parallel (bool): Whether multi-cards training. Default: True.
        validate (bool): Whether to do evaluation. Default: False.

    """
    if use_fleet:
        fleet.init(is_collective=True)

    logger = get_logger("paddlevideo")
    batch_size = cfg.DATASET.get('batch_size', 8)
    valid_batch_size = cfg.DATASET.get('valid_batch_size', batch_size)

    use_gradient_accumulation = cfg.get('GRADIENT_ACCUMULATION', None)
    if use_gradient_accumulation and dist.get_world_size() >= 1:
        global_batch_size = cfg.GRADIENT_ACCUMULATION.get(
            'global_batch_size', None)
        num_gpus = dist.get_world_size()

        assert isinstance(
            global_batch_size, int
        ), f"global_batch_size must be int, but got {type(global_batch_size)}"
        assert batch_size < global_batch_size, f"global_batch_size must bigger than batch_size"

        cur_global_batch_size = batch_size * num_gpus  # The number of batches calculated by all GPUs at one time
        assert global_batch_size % cur_global_batch_size == 0, \
            f"The global batchsize must be divisible by cur_global_batch_size, but \
                {global_batch_size} % {cur_global_batch_size} != 0"

        cfg.GRADIENT_ACCUMULATION[
            "num_iters"] = global_batch_size // cur_global_batch_size
        # The number of iterations required to reach the global batchsize
        logger.info(
            f"Using gradient accumulation training strategy, "
            f"global_batch_size={global_batch_size}, "
            f"num_gpus={num_gpus}, "
            f"num_accumulative_iters={cfg.GRADIENT_ACCUMULATION.num_iters}")

    places = paddle.set_device('gpu')

    # default num worker: 0, which means no subprocess will be created
    num_workers = cfg.DATASET.get('num_workers', 0)
    valid_num_workers = cfg.DATASET.get('valid_num_workers', num_workers)
    model_name = cfg.model_name
    output_dir = cfg.get("output_dir", f"./output/{model_name}")
    mkdir(output_dir)

    writer = LogWriter(logdir=output_dir)
    writer_cs = LogWriter(logdir=os.path.join(output_dir, "class_scores"))

    # 1. Construct model
    model = build_model(cfg.MODEL)
    #print(model)
    if parallel:
        model = paddle.DataParallel(model)

    if use_fleet:
        model = paddle.distributed_model(model)

    # 2. Construct dataset and dataloader
    train_dataset = build_dataset((cfg.DATASET.train, cfg.PIPELINE.train))
    train_dataloader_setting = dict(batch_size=batch_size,
                                    num_workers=num_workers,
                                    collate_fn_cfg=cfg.get('MIX', None),
                                    places=places)

    train_loader = build_dataloader(train_dataset, **train_dataloader_setting)
    if validate:
        valid_dataset = build_dataset((cfg.DATASET.valid, cfg.PIPELINE.valid))
        validate_dataloader_setting = dict(
            batch_size=valid_batch_size,
            num_workers=valid_num_workers,
            places=places,
            drop_last=False,
            shuffle=cfg.DATASET.get(
                'shuffle_valid',
                False)  #NOTE: attention lstm need shuffle valid data.
        )
        valid_loader = build_dataloader(valid_dataset,
                                        **validate_dataloader_setting)

    # 3. Construct solver.
    lr = build_lr(cfg.OPTIMIZER.learning_rate, len(train_loader))
    optimizer = build_optimizer(cfg.OPTIMIZER,
                                lr,
                                parameter_list=model.parameters())
    if use_fleet:
        optimizer = fleet.distributed_optimizer(optimizer)
    # Resume
    resume_epoch = cfg.get("resume_epoch", 0)
    if resume_epoch:
        filename = osp.join(output_dir,
                            model_name + f"_epoch_{resume_epoch:05d}")
        resume_model_dict = load(filename + '.pdparams')
        resume_opt_dict = load(filename + '.pdopt')
        model.set_state_dict(resume_model_dict)
        optimizer.set_state_dict(resume_opt_dict)

    # Finetune:
    if weights:
        assert resume_epoch == 0, f"Conflict occurs when finetuning, please switch resume function off by setting resume_epoch to 0 or not indicating it."
        model_dict = load(weights)
        model.set_state_dict(model_dict)

    # 4. Train Model
    ###AMP###
    if amp:
        scaler = paddle.amp.GradScaler(init_loss_scaling=2.0**16,
                                       incr_every_n_steps=2000,
                                       decr_every_n_nan_or_inf=1)

    best = 0.
    for epoch in range(0, cfg.epochs):
        if epoch < resume_epoch:
            logger.info(
                f"| epoch: [{epoch+1}] <= resume_epoch: [{ resume_epoch}], continue... "
            )
            continue
        model.train()

        ###############################
        #print(model)
        #for na, param in model.named_parameters():
        #    print(na, param.shape)
        """
        if epoch==20 or resume_epoch!=0:
            resume_epoch = 0
            for na, param in model.named_parameters():
                if "PA" in na:
                    print(param.stop_gradient)
                    param.stop_gradient = False
            for na, param in model.named_parameters():
                if "PA" in na:
                    print(param.stop_gradient)
        """
        ##################################

        record_list = build_record(cfg.MODEL)
        tic = time.time()
        for i, data in enumerate(train_loader):
            record_list['reader_time'].update(time.time() - tic)

            # 4.1 forward

            #####1119 R-drop
            """
            data[0] = paddle.concat([data[0], data[0]], axis=0)
            if len(data[1:])==3:
                data[1] = paddle.concat([data[1], data[1]], axis=0)
                data[2] = paddle.concat([data[2], data[2]], axis=0)
            #print(data[0].shape, data[1].shape, data[2].shape)
            """
            #####

            ###AMP###
            if amp:
                with paddle.amp.auto_cast(custom_black_list={"reduce_mean"}):
                    outputs = model(data, mode='train')

                avg_loss = outputs['loss']
                scaled = scaler.scale(avg_loss)
                scaled.backward()
                # keep prior to 2.0 design
                scaler.minimize(optimizer, scaled)
                optimizer.clear_grad()

            else:
                outputs = model(data, mode='train')

                # 4.2 backward
                if use_gradient_accumulation and i == 0:  # Use gradient accumulation strategy
                    optimizer.clear_grad()
                avg_loss = outputs['loss']
                avg_loss.backward()

                # 4.3 minimize
                if use_gradient_accumulation:  # Use gradient accumulation strategy
                    if (i + 1) % cfg.GRADIENT_ACCUMULATION.num_iters == 0:
                        for p in model.parameters():
                            p.grad.set_value(
                                p.grad / cfg.GRADIENT_ACCUMULATION.num_iters)
                        optimizer.step()
                        optimizer.clear_grad()
                else:  # Common case
                    optimizer.step()
                    optimizer.clear_grad()

            # log record
            record_list['lr'].update(optimizer.get_lr(), batch_size)
            for name, value in outputs.items():
                record_list[name].update(value, batch_size)

            record_list['batch_time'].update(time.time() - tic)
            tic = time.time()

            if i % cfg.get("log_interval", 10) == 0:
                ips = "ips: {:.5f} instance/sec.".format(
                    batch_size / record_list["batch_time"].val)
                log_batch(record_list, i, epoch + 1, cfg.epochs, "train", ips)

                writer.add_scalar('train/loss', avg_loss, i+epoch*len(train_loader))
                writer.add_scalar('train/lr', optimizer.get_lr(), i+epoch*len(train_loader))
                writer.add_scalar('train/acc', record_list['top1'].avg, i+epoch*len(train_loader))

            # learning rate iter step
            if cfg.OPTIMIZER.learning_rate.get("iter_step"):
                lr.step()

        # learning rate epoch step
        if not cfg.OPTIMIZER.learning_rate.get("iter_step"):
            lr.step()

        ips = "avg_ips: {:.5f} instance/sec.".format(
            batch_size * record_list["batch_time"].count /
            record_list["batch_time"].sum)
        log_epoch(record_list, epoch + 1, "train", ips)

        def evaluate(best):
            model.eval()
            record_list = build_record(cfg.MODEL)
            record_list.pop('lr')
            tic = time.time()
            cls_score_list = []
            label_list = []
            for i, data in enumerate(valid_loader):
                outputs, cls_score, label = model(data, mode='valid')
                cls_score_list.append(cls_score)
                label_list.append(label)

                # log_record
                for name, value in outputs.items():
                    record_list[name].update(value, batch_size)

                record_list['batch_time'].update(time.time() - tic)
                tic = time.time()

                if i % cfg.get("log_interval", 10) == 0:
                    ips = "ips: {:.5f} instance/sec.".format(
                        batch_size / record_list["batch_time"].val)
                    log_batch(record_list, i, epoch + 1, cfg.epochs, "val", ips)

            ips = "avg_ips: {:.5f} instance/sec.".format(
                batch_size * record_list["batch_time"].count /
                record_list["batch_time"].sum)
            log_epoch(record_list, epoch + 1, "val", ips)

            best_flag = False
            for top_flag in ['hit_at_one', 'top1']:
                if record_list.get(
                        top_flag) and record_list[top_flag].avg > best:
                    best = record_list[top_flag].avg
                    best_flag = True
            
            cls_scores = paddle.concat(cls_score_list, axis=0)
            labels = paddle.concat(label_list, axis=0)
            scores_df = []
            
            #统计并展示每个类别的acc
            for i in range(30):
                index = paddle.to_tensor(np.argwhere(labels.numpy().reshape(-1)==i).reshape(-1))
                #print(index)
                top1 = paddle.metric.accuracy(input=cls_scores[index], label=labels[index], k=1)
                top5 = paddle.metric.accuracy(input=cls_scores[index], label=labels[index], k=5)
                tag = ["label-"+str(i)+"/top1", "label-"+str(i)+"/top5"]
                writer_cs.add_scalar(tag=tag[0], value=top1, step=epoch)
                writer_cs.add_scalar(tag=tag[1], value=top5, step=epoch)
                scores_df.append([len(index), top1.numpy().item(), top5.numpy().item()])
            scores_df = pd.DataFrame(scores_df, columns=["num", "top1", "top5"])

            total_score = paddle.mean((paddle.argmax(cls_scores, axis=-1)==paddle.reshape(labels, shape=(-1,))).astype("float32")).numpy().item()
            print(f"total_score: {total_score}, class scores:\n", scores_df)

            return best, best_flag, record_list[top_flag].avg, record_list["loss"].avg

        # use precise bn to improve acc
        if cfg.get("PRECISEBN") and (epoch % cfg.PRECISEBN.preciseBN_interval
                                     == 0 or epoch == cfg.epochs - 1):
            do_preciseBN(
                model, train_loader, parallel,
                min(cfg.PRECISEBN.num_iters_preciseBN, len(train_loader)))

        # 5. Validation
        if validate and (epoch % cfg.get("val_interval", 1) == 0
                         or epoch == cfg.epochs - 1):
            with paddle.no_grad():
                best, save_best_flag, now_acc, now_loss = evaluate(best)
                writer.add_scalar('val/acc', now_acc, epoch)
                writer.add_scalar('val/loss', now_loss, epoch)
            # save best
            if save_best_flag:
                save(optimizer.state_dict(),
                     osp.join(output_dir, model_name + "_best.pdopt"))
                save(model.state_dict(),
                     osp.join(output_dir, model_name + "_best.pdparams"))
                if model_name == "AttentionLstm":
                    logger.info(
                        f"Already save the best model (hit_at_one){best}")
                else:
                    logger.info(
                        f"Already save the best model (top1 acc){int(best *10000)/10000}"
                    )

        # 6. Save model and optimizer
        if epoch % cfg.get("save_interval", 1) == 0 or epoch == cfg.epochs - 1:
            save(
                optimizer.state_dict(),
                osp.join(output_dir,
                         model_name + f"_epoch_{epoch+1:05d}.pdopt"))
            save(
                model.state_dict(),
                osp.join(output_dir,
                         model_name + f"_epoch_{epoch+1:05d}.pdparams"))

    logger.info(f'training {model_name} finished')
