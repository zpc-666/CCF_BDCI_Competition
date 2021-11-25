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

import paddle
from paddlevideo.utils import get_logger
from ..loader.builder import build_dataloader, build_dataset
from ..metrics import build_metric
from ..modeling.builder import build_model
from paddlevideo.utils import load
import copy

logger = get_logger("paddlevideo")

inward_ori_index = [(1, 8), (0, 1), (15, 0), (17, 15), (16, 0),
                    (18, 16), (5, 1), (6, 5), (7, 6), (2, 1), (3, 2),
                    (4, 3), (9, 8), (10, 9), (11, 10), (24, 11),
                    (22, 11), (23, 22), (12, 8), (13, 12), (14, 13),
                    (21, 14), (19, 14), (20, 19)]

def get_bone(joint_data):
    bone_data = paddle.zeros_like(joint_data)
    for v1, v2 in inward_ori_index:
        bone_data[:, :, :, v1, :] = joint_data[:, :, :, v1, :]-joint_data[:, :, :, v2, :]
    return bone_data

def get_joint_motion(joint_data):
    motion_data = paddle.zeros_like(joint_data)
    frame_len = motion_data.shape[2]
    motion_data[:, :, :frame_len-1, :, :] = joint_data[:, :, 1:frame_len, :, :]-joint_data[:, :, :frame_len-1, :, :]
    return motion_data

@paddle.no_grad()
def test_model(cfg, weights, parallel=True):
    """Test model entry

    Args:
        cfg (dict): configuration.
        weights (str): weights path to load.
        parallel (bool): Whether to do multi-cards testing. Default: True.

    """
    # 1. Construct model.
    if cfg.MODEL.backbone.get('pretrained'):
        cfg.MODEL.backbone.pretrained = ''  # disable pretrain model init

    weights = [weights]+["output/AGCN/ctrgcn_dr05_leps03_ks9_nofreeze_floss_bs8/AGCN_best138.pdparams", "output/AGCN/ctrgcn_dr05_leps03_ks9_nofreeze_floss_jm/AGCN_best163.pdparams", \
            "output/AGCN/ctrgcn_dr05_leps03_ks9_nofreeze_floss_b/AGCN_best.pdparams", "output/AGCN/ctrgcn_dr05_leps03_ks9_nofreeze_floss_bm/AGCN_best164.pdparams"]

    models = []
    for w in weights:
        state_dicts = load(w)
        models.append(build_model(cfg.MODEL))
        models[-1].eval()
        models[-1].set_state_dict(state_dicts)

    #if parallel:
    #    model = paddle.DataParallel(model)

    # 2. Construct dataset and dataloader.
    cfg.DATASET.test.test_mode = True
    dataset = build_dataset((cfg.DATASET.test, cfg.PIPELINE.test))
    batch_size = cfg.DATASET.get("test_batch_size", 8)
    places = paddle.set_device('gpu')
    # default num worker: 0, which means no subprocess will be created
    num_workers = cfg.DATASET.get('num_workers', 0)
    num_workers = cfg.DATASET.get('test_num_workers', num_workers)
    dataloader_setting = dict(batch_size=batch_size,
                              num_workers=num_workers,
                              places=places,
                              drop_last=False,
                              shuffle=False)

    data_loader = build_dataloader(dataset, **dataloader_setting)

    #model.eval()

    # add params to metrics
    cfg.METRIC.data_size = len(dataset)
    cfg.METRIC.batch_size = batch_size

    is_joint = ['j', 'j', 'jm', 'b', 'bm']
    assert len(is_joint)==len(models)
    
    Metric = build_metric(cfg.METRIC)
    for batch_id, data in enumerate(data_loader):
        outputs = 0.
        for mi, model in enumerate(models):
            if is_joint[mi]=='b':
                new_data = copy.deepcopy(data)
                new_data[0] = get_bone(new_data[0])
                pred = model(new_data, mode='test')
            elif is_joint[mi]=='jm':
                new_data = copy.deepcopy(data)
                new_data[0] = get_joint_motion(new_data[0])
                pred = model(new_data, mode='test')
            elif is_joint[mi]=='bm':
                new_data = copy.deepcopy(data)
                new_data[0] = get_bone(new_data[0])
                new_data[0] = get_joint_motion(new_data[0])
                pred = model(new_data, mode='test')
            else:
                pred = model(data, mode='test')
            outputs += pred/len(models)
        Metric.update(batch_id, data, outputs)
    Metric.accumulate()
