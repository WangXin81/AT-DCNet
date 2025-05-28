import os
# import torch
import numpy as np
from datasets import dataloader_utils as utils
# from misc.utils import save_image
import matplotlib.pyplot as plt
from models.networks import *



class CDEvaluator:
    def __init__(self, args):

        self.n_class = args.n_class
        # define G
        self.net_G = define_G(args=args, gpu_ids=args.gpu_ids)
        self.args_net_G = args.net_G
        self.device = torch.device("cuda:%s" % args.gpu_ids[0]
                                   if torch.cuda.is_available() and len(args.gpu_ids) > 0
                                   else "cpu")

        print(self.device)
        self.checkpoint_dir = args.checkpoint_dir
        self.project_name = args.project_name
        self.pred_dir = args.output_folder
        os.makedirs(self.pred_dir, exist_ok=True)

    def load_checkpoint(self, checkpoint_name='best_ckpt.pt'):

        if os.path.exists(os.path.join(self.checkpoint_dir, checkpoint_name)):
            # load the entire checkpoint
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, checkpoint_name),
                                    map_location=self.device)

            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])
            self.net_G.to(self.device)
            # update some other states
            self.best_val_mF1 = checkpoint['best_val_mF1']
            self.best_epoch_id = checkpoint['best_epoch_id']

        else:
            raise FileNotFoundError('no such checkpoint %s' % checkpoint_name)
        return self.net_G

    def _visualize_pred(self):
        pred = torch.argmax(self.G_pred, dim=1, keepdim=True)
        pred = pred * 255
        return pred

    def _forward_pass(self, batch, name):
        self.batch = batch
        img_in1 = batch['A'].to(self.device)
        img_in2 = batch['B'].to(self.device)
        self.shape_h = img_in1.shape[-2]
        self.shape_w = img_in1.shape[-1]
        if self.args_net_G == 'DMINet':
            self.G_pred1, self.G_pred2, self.G_middle1, self.G_middle2 = self.net_G(img_in1, img_in2)
            self.G_pred = self.G_pred1 + self.G_pred2
        elif self.args_net_G == 'ChangeFormerV5':
            self.G_pred = self.net_G(img_in1, img_in2)
            self.G_pred = self.G_pred[-1]
        elif self.args_net_G == 'ICIFNet':
            self.G_pred1, self.G_pred2, self.G_pred3 = self.net_G(img_in1, img_in2)
            self.G_pred = self.G_pred1 + self.G_pred2 + self.G_pred3
        else:
            self.G_pred = self.net_G(img_in1, img_in2 , self.project_name, name)
        return self._visualize_pred()

    def eval(self):
        self.net_G.eval()

    def _save_predictions(self):
        """
        保存模型输出结果，二分类图像
        """
        # target = self.batch['L'].to(self.device).detach()
        preds = self._visualize_pred()
        name = self.batch['name']
        for i, pred in enumerate(preds):
            file_name = os.path.join(
                self.pred_dir, self.project_name, name[i].replace('.jpg', '.png'))
            os.makedirs(os.path.join(self.pred_dir, self.project_name), exist_ok=True)
            pred = pred[0].cpu().numpy()
            save_image(pred, file_name)
            # imsave1(file_name, pred, target)

        # vis_pred = utils.make_numpy_grid(self._visualize_pred())
        # vis = np.concatenate([vis_input, vis_input2, vis_pred, vis_gt], axis=0)

        # target = self.batch['L'].to(self.device).detach()
        # vis = np.clip(self._visualize_pred().cpu(), a_min=0.0, a_max=1.0)
        # name = self.batch['name']
        # file_name = os.path.join(
        #         self.pred_dir, self.project_name, name[0].replace('.jpg', '.png'))
        # os.makedirs(os.path.join(self.pred_dir, self.project_name), exist_ok=True)
        # imsave1(file_name, target, vis)
