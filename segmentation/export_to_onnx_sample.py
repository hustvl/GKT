import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import argparse
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt

import tqdm
import torch

from data.dataset import semantic_dataset
from data.const import NUM_CLASSES
from model import get_model
from postprocess.vectorize import vectorize


def export_onnx(model, val_loader, onnx_save_path, model_mode, model_load_path, is_withbatch):
    model_name = model_load_path.split("/")[-1]
    if model_name[-3:]==".pt":
        model_replace_name = model_name[:-3] + "_v1.1.onnx"
    else:
        model_replace_name = "model.onnx"
    onnx_model_path = os.path.join(onnx_save_path, model_mode+"_"+model_replace_name)
    print("onnx_model_path: ", onnx_model_path)
    model.eval()
    with torch.no_grad():
        for batchi, (imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, semantic_gt, instance_gt, direction_gt) in enumerate(val_loader):
            print("=-"*30)
            print("imgs.shape: ", imgs.shape)
            print("trans.shape: ", trans.shape)
            print("rots.shape: ", rots.shape)
            print("intrins.shape: ", intrins.shape)
            print("post_trans.shape: ", post_trans.shape)
            print("post_rots.shape: ", post_rots.shape)
            print("lidar_data.shape: ", lidar_data.shape)
            print("lidar_mask.shape: ", lidar_mask.shape)
            print("car_trans.shape: ", car_trans.shape)
            print("yaw_pitch_roll.shape: ", yaw_pitch_roll.shape)
            if is_withbatch:
                dummy_input = tuple([imgs.cuda(),            # torch.Size([4, 6, 3, 128, 352])
                                trans.cuda(),                # torch.Size([4, 6, 3])
                                rots.cuda(),                 # torch.Size([4, 6, 3, 3])
                                intrins.cuda(),              # torch.Size([4, 6, 3, 3])
                                post_trans.cuda(),           # torch.Size([4, 6, 3])
                                post_rots.cuda(),            # torch.Size([4, 6, 3, 3])
                                lidar_data.cuda(),           # torch.Size([4, 81920, 5])
                                lidar_mask.cuda(),           # torch.Size([4, 81920])
                                car_trans.cuda(),            # torch.Size([4, 3])
                                yaw_pitch_roll.cuda()])      # torch.Size([4, 3])
                break
            else:
                dummy_input = tuple([imgs[0].cuda(),         # torch.Size([6, 3, 128, 352])
                                trans[0].cuda(),             # torch.Size([6, 3])
                                rots[0].cuda(),              # torch.Size([6, 3, 3])
                                intrins[0].cuda(),           # torch.Size([6, 3, 3])
                                post_trans[0].cuda(),        # torch.Size([6, 3])
                                post_rots[0].cuda(),         # torch.Size([6, 3, 3])
                                lidar_data[0].cuda(),        # torch.Size([81920, 5])
                                lidar_mask[0].cuda(),        # torch.Size([81920])
                                car_trans[0].cuda(),         # torch.Size([3])
                                yaw_pitch_roll[0].cuda()])   # torch.Size([3])
                break
        # torch.onnx.export(model, args = dummy_input, f = onnx_model_path,
        #                   verbose=True, opset_version=11,
        #                   input_names=['imgs', 'trans', 'rots', 'intrins', 'post_trans', 'post_rots', 'lidar_data', 'lidar_mask', 'car_trans', 'yaw_pitch_roll'],
        #                   output_names=["semantic", "embedding", "direction"])

        torch.onnx.export(model, args=dummy_input, f=onnx_model_path,
                          verbose=True, opset_version=11,
                          input_names=['imgs', 'trans', 'rots', 'intrins', 'post_trans', 'post_rots', 'lidar_data',
                                       'lidar_mask', 'car_trans', 'yaw_pitch_roll'],
                          output_names=['semantic'])

        # torch.onnx.export(model, args=dummy_input, f=onnx_model_path,
        #                   verbose=True, opset_version=11,
        #                   input_names=['imgs', 'trans', 'rots', 'intrins', 'post_trans', 'post_rots', 'lidar_data',
        #                                'lidar_mask', 'car_trans', 'yaw_pitch_roll'],
        #                   output_names=['semantic', 'instance', 'direction'])

        print("convert done!")


def main(args):
    data_conf = {
        'num_channels': NUM_CLASSES + 1,
        'image_size': args.image_size,
        'xbound': args.xbound,
        'ybound': args.ybound,
        'zbound': args.zbound,
        'dbound': args.dbound,
        'thickness': args.thickness,
        'angle_class': args.angle_class,
    }
    print("onnx_save_path: ", args.onnx_save_path)
    print("model: ", args.model)
    print("modelf: ", args.modelf)
    print("is_withbatch: ", args.is_withbatch)
    _, val_loader = semantic_dataset(args.version, args.dataroot, data_conf, args.bsz, args.nworkers)
    # model = get_model(args.model, data_conf, args.instance_seg, args.embedding_dim, args.direction_pred, args.angle_class, args.is_withbatch)
    model = get_model(args.model, data_conf, args.instance_seg, args.embedding_dim, args.direction_pred,
                      args.angle_class)
    # model = get_model(arg)
    model.load_state_dict(torch.load(args.modelf), strict=False)
    model.cuda()
    export_onnx(model, val_loader, args.onnx_save_path, args.model, args.modelf, args.is_withbatch)



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    # logging config
    parser.add_argument("--logdir", type=str, default='./runs')

    # nuScenes config
    parser.add_argument('--dataroot', type=str, default='dataset/nuScenes/')
    parser.add_argument('--version', type=str, default='v1.0-mini', choices=['v1.0-trainval', 'v1.0-mini'])

    # model config
    parser.add_argument("--model", type=str, default='HDMapNet_cam')   # choose in ["lift_splat", "HDMapNet_cam", "HDMapNet_lidar", HDMapNet_fusion"]

    # training config
    parser.add_argument("--nepochs", type=int, default=30)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--pos_weight", type=float, default=2.13)
    parser.add_argument("--bsz", type=int, default=1)
    parser.add_argument("--nworkers", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-7)

    # finetune config
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--modelf', type=str, default=None)

    # data config
    parser.add_argument("--thickness", type=int, default=5)
    parser.add_argument("--image_size", nargs=2, type=int, default=[128, 352])
    parser.add_argument("--xbound", nargs=3, type=float, default=[-30.0, 30.0, 0.15])
    parser.add_argument("--ybound", nargs=3, type=float, default=[-15.0, 15.0, 0.15])
    parser.add_argument("--zbound", nargs=3, type=float, default=[-10.0, 10.0, 20.0])
    parser.add_argument("--dbound", nargs=3, type=float, default=[4.0, 45.0, 1.0])

    # embedding config
    parser.add_argument('--instance_seg', action='store_true')
    parser.add_argument("--embedding_dim", type=int, default=16)
    parser.add_argument("--delta_v", type=float, default=0.5)
    parser.add_argument("--delta_d", type=float, default=3.0)

    # direction config
    parser.add_argument('--direction_pred', action='store_true')
    parser.add_argument('--angle_class', type=int, default=36)

    # loss config
    parser.add_argument("--scale_seg", type=float, default=1.0)
    parser.add_argument("--scale_var", type=float, default=1.0)
    parser.add_argument("--scale_dist", type=float, default=1.0)
    parser.add_argument("--scale_direction", type=float, default=0.2)
    
    # onnx para
    parser.add_argument("--onnx_save_path", type=str, default=None)
    parser.add_argument("--is_withbatch", action='store_true')

    args = parser.parse_args()
    main(args)





