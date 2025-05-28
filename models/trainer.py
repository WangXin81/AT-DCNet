import numpy as np
import os
from models.networks import *
from misc.utils import get_scheduler
import torch
import torch.optim as optim
from misc.metric_tool import ConfuseMatrixMeter
from misc.losses import cross_entropy, CombinedLoss, WeightedBCELoss
from misc.logger_tool import Logger, Timer
import time
CUDA_LAUNCH_BLOCKING=1

class CDTrainer:

    def __init__(self, args, dataloaders):

        self.dataloaders = dataloaders
        self.n_class = args.n_class
        self.args_net_G = args.net_G
        self.weight1 = args.weight1
        self.weight2 = args.weight2
        self.gamma = args.gamma
        # define G
        self.net_G = define_G(args=args, gpu_ids=args.gpu_ids)
        self.device = torch.device("cuda:%s" % args.gpu_ids[0] if torch.cuda.is_available() and len(args.gpu_ids) > 0
                                   else "cpu")
        print(self.device)
        # Learning rate and Beta1 for Adam optimizers
        self.lr = args.lr
        if args.optimizer == 'adam':
            self.optimizer_G = torch.optim.Adam(self.net_G.parameters(), self.lr, weight_decay=5e-4)
        elif args.optimizer == 'adamW':
            self.optimizer_G = torch.optim.AdamW(self.net_G.parameters(), self.lr, weight_decay=5e-4)
        elif args.optimizer == 'sgd':
            self.optimizer_G = optim.SGD(self.net_G.parameters(), lr=self.lr,
                                         momentum=0.9,
                                         weight_decay=5e-4)
        else:
            raise NotImplementedError
        # scheduler = get_scheduler(optimizer, len(train_loader), opt)
        # define lr schedulers
        self.exp_lr_scheduler_G = get_scheduler(self.optimizer_G, args)
        self.running_metric = ConfuseMatrixMeter(n_class=2)

        # define logger file
        logger_path = os.path.join(args.checkpoint_dir, 'log.txt')
        self.logger = Logger(logger_path)
        self.logger.write_dict_str(args.__dict__)
        # define timer
        self.timer = Timer()
        self.batch_size = args.batch_size

        #  training log
        self.epoch_mF1 = 0
        self.best_val_mF1 = 0.0
        self.best_epoch_id = 0
        self.epoch_to_start = 0
        self.max_num_epochs = args.max_epochs

        self.global_step = 0
        self.steps_per_epoch = len(dataloaders['train'])
        self.total_steps = (self.max_num_epochs - self.epoch_to_start) * self.steps_per_epoch

        self.G_pred = None
        if args.net_G == 'DMINet':
            self.G_pred1 = None
            self.G_pred2 = None
            self.G_pred3 = None

        self.pred_vis = None
        self.batch = None
        self.G_loss = None
        self.is_training = False
        self.batch_id = 0
        self.epoch_id = 0
        self.epoch_loss = 0
        self.checkpoint_dir = args.checkpoint_dir
        self.vis_dir = args.vis_dir

        # define the loss functions
        if args.loss == 'ce':
            self._pxl_loss = cross_entropy
        elif args.loss == 'bcew':
            self._pxl_loss = WeightedBCELoss(self.weight1, self.weight2, self.gamma)
        elif args.loss == 'hybridLoss':
            self._pxl_loss = CombinedLoss()
        else:
            raise NotImplemented(args.loss)

        self.VAL_mF1 = np.array([], np.float32)
        if os.path.exists(os.path.join(self.checkpoint_dir, 'val_mF1.npy')):
            self.VAL_mF1 = np.load(os.path.join(self.checkpoint_dir, 'val_mF1.npy'))
        self.TRAIN_mF1 = np.array([], np.float32)
        if os.path.exists(os.path.join(self.checkpoint_dir, 'train_mF1.npy')):
            self.TRAIN_mF1 = np.load(os.path.join(self.checkpoint_dir, 'train_mF1.npy'))

        self.VAL_loss = np.array([], np.float32)
        if os.path.exists(os.path.join(self.checkpoint_dir, 'val_loss.npy')):
            self.VAL_loss = np.load(os.path.join(self.checkpoint_dir, 'val_loss.npy'))
        self.TRAIN_loss = np.array([], np.float32)
        if os.path.exists(os.path.join(self.checkpoint_dir, 'train_loss.npy')):
            self.TRAIN_loss = np.load(os.path.join(self.checkpoint_dir, 'train_loss.npy'))

        # check and create model dir
        if os.path.exists(self.checkpoint_dir) is False:
            os.mkdir(self.checkpoint_dir)
        if os.path.exists(self.vis_dir) is False:
            os.mkdir(self.vis_dir)

    def _load_checkpoint(self, ckpt_name='last_ckpt.pt'):

        if os.path.exists(os.path.join(self.checkpoint_dir, ckpt_name)):
            self.logger.write('loading last checkpoint...\n')
            # load the entire checkpoint
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, ckpt_name),
                                    map_location=self.device)
            # update net_G states
            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])

            self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            self.exp_lr_scheduler_G.load_state_dict(checkpoint['exp_lr_scheduler_G_state_dict'])

            self.net_G.to(self.device)
            # print(self.device)

            # update some other states
            self.epoch_to_start = checkpoint['epoch_id'] + 1
            self.best_val_mF1 = checkpoint['best_val_mF1']
            self.best_epoch_id = checkpoint['best_epoch_id']

            self.total_steps = (self.max_num_epochs - self.epoch_to_start) * self.steps_per_epoch

            self.logger.write('Epoch_to_start = %d, Historical_best_mF1 = %.4f (at epoch %d)\n' %
                              (self.epoch_to_start, self.best_val_mF1, self.best_epoch_id))
            self.logger.write('\n')

        else:
            print('training from scratch...')

    def _timer_update(self):
        self.global_step = (self.epoch_id - self.epoch_to_start) * self.steps_per_epoch + self.batch_id

        self.timer.update_progress((self.global_step + 1) / self.total_steps)
        est = self.timer.estimated_remaining()  # 剩余时间
        imps = (self.global_step + 1) * self.batch_size / self.timer.get_stage_elapsed()  # 单位时间内处理的图像数
        return imps, est

    def _visualize_pred(self):
        pred = torch.argmax(self.G_pred, dim=1, keepdim=True)
        pred_vis = pred * 255
        return pred_vis

    def _save_checkpoint(self, ckpt_name):
        checkpoint_path = os.path.join(self.checkpoint_dir, ckpt_name)
        torch.save({
            'epoch_id': self.epoch_id,
            'best_val_mF1': self.best_val_mF1,
            'best_epoch_id': self.best_epoch_id,
            'model_G_state_dict': self.net_G.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'exp_lr_scheduler_G_state_dict': self.exp_lr_scheduler_G.state_dict(),
        }, checkpoint_path)

    def _update_lr_schedulers(self):
        self.exp_lr_scheduler_G.step()

    def _update_metric(self):
        """
        update metric
        """
        target = self.batch['L'].to(self.device).detach()
        G_pred = self.G_pred.detach()
        G_pred = torch.argmax(G_pred, dim=1)
        current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy(), gt=target.cpu().numpy())
        return current_score

    def _collect_running_batch_states(self):

        running_mF1 = self._update_metric()
        m = len(self.dataloaders['train'])
        if self.is_training is False:
            m = len(self.dataloaders['val'])

        imps, est = self._timer_update()
        if np.mod(self.batch_id, 1000) == 1:
            message = 'Is_training: %s. [%d,%d][%d,%d], imps: %.2f, est: %.2fh, G_loss: %.5f, running_mf1: %.5f\n' % \
                      (self.is_training, self.epoch_id, self.max_num_epochs - 1, self.batch_id, m,
                       imps * self.batch_size, est,
                       self.G_loss.item(), running_mF1)
            self.logger.write(message)

    def _collect_epoch_states(self, epoch_loss):
        scores = self.running_metric.get_scores()
        self.epoch_mF1 = scores['mf1']
        self.epoch_loss = epoch_loss
        self.logger.write('Is_training: %s. Epoch %d / %d, epoch_mF1= %.5f, epoch_loss=%.5f\n' %
                          (self.is_training, self.epoch_id, self.max_num_epochs - 1, self.epoch_mF1, epoch_loss))
        message = ''
        for k, v in scores.items():
            message += '%s: %.5f ' % (k, v)
        self.logger.write(message + '\n')
        self.logger.write('\n')

    def _update_checkpoints(self):

        # save current model
        self._save_checkpoint(ckpt_name='last_ckpt.pt')
        self.logger.write('Lastest model updated. Epoch_mF1=%.4f, Historical_best_mF1=%.4f (at epoch %d)\n'
                          % (self.epoch_mF1, self.best_val_mF1, self.best_epoch_id))
        self.logger.write('\n')

        # update the best model (based on eval acc)
        if self.epoch_mF1 > self.best_val_mF1:
            self.best_val_mF1 = self.epoch_mF1
            self.best_epoch_id = self.epoch_id
            self._save_checkpoint(ckpt_name='best_ckpt.pt')
            self.logger.write('*' * 10 + 'Best model updated!\n')
            self.logger.write('\n')

    def _update_training_curve(self):
        # update train acc curve
        self.TRAIN_mF1 = np.append(self.TRAIN_mF1, [self.epoch_mF1])
        np.save(os.path.join(self.checkpoint_dir, 'train_mF1.npy'), self.TRAIN_mF1)
        self.TRAIN_loss = np.append(self.TRAIN_loss, [self.epoch_loss])
        np.save(os.path.join(self.checkpoint_dir, 'train_loss.npy'), self.TRAIN_loss)

    def _update_val_curve(self):
        # update val acc curve
        self.VAL_mF1 = np.append(self.VAL_mF1, [self.epoch_mF1])
        np.save(os.path.join(self.checkpoint_dir, 'val_mF1.npy'), self.VAL_mF1)
        self.VAL_loss = np.append(self.VAL_loss, [self.epoch_loss])
        np.save(os.path.join(self.checkpoint_dir, 'val_loss.npy'), self.VAL_loss)

    def _clear_cache(self):
        self.running_metric.clear()

    def _forward_pass(self, batch):
        self.batch = batch
        img_in1 = batch['A'].to(self.device)
        img_in2 = batch['B'].to(self.device)
        if self.args_net_G == 'DMINet':
            self.G_pred1, self.G_pred2, self.G_middle1, self.G_middle2 = self.net_G(img_in1, img_in2)
            self.G_pred = self.G_pred1 + self.G_pred2
        elif self.args_net_G == 'ICIFNet':
            self.G_pred1, self.G_pred2, self.G_pred3 = self.net_G(img_in1, img_in2)
            self.G_pred = self.G_pred1 + self.G_pred2 + self.G_pred3
        elif self.args_net_G == 'ChangeFormerV5':
            self.G_pred = self.net_G(img_in1, img_in2)
            self.G_pred = self.G_pred[-1]
        else:
            self.G_pred = self.net_G(img_in1, img_in2)

    def _backward_G(self):
        gt = self.batch['L'].to(self.device).long()
        # gt = torch.cat([1 - gt, gt], dim=1).to(self.device).float()
        if self.args_net_G == 'DMINet':
            self.G_loss = self._pxl_loss(self.G_pred1, gt) + self._pxl_loss(self.G_pred2, gt) + 0.5 * (
                    self._pxl_loss(self.G_middle1, gt) + self._pxl_loss(self.G_middle2, gt))
        elif self.args_net_G == 'ICIFNet':
            self.G_loss = self._pxl_loss(self.G_pred1, gt) + self._pxl_loss(self.G_pred2, gt) + 0.5 * self._pxl_loss(
                self.G_pred3, gt)
        else:
            self.G_loss = self._pxl_loss(self.G_pred, gt)
        self.G_loss.backward()

    def train_models(self):

        self._load_checkpoint()
        # loop over the dataset multiple times
        for self.epoch_id in range(self.epoch_to_start, self.max_num_epochs):
            # ################# train #################
            self._clear_cache()
            train_epoch_loss = 0
            val_epoch_loss = 0
            self.is_training = True
            starttime = time.time()
            self.net_G.train()  # Set model to training mode
            # Iterate over data.
            self.logger.write('lr: %0.7f\n' % self.optimizer_G.param_groups[0]['lr'])
            for self.batch_id, batch in enumerate(self.dataloaders['train'], 0):
                self._forward_pass(batch)
                # update G
                self.optimizer_G.zero_grad()
                self._backward_G()
                train_epoch_loss += self.G_loss.item()
                self.optimizer_G.step()
                self._collect_running_batch_states()
                self._timer_update()

            average_train_loss = train_epoch_loss / len(self.dataloaders['train'])
            self._collect_epoch_states(average_train_loss)
            self._update_training_curve()
            self._update_lr_schedulers()

            endtime = time.time() - starttime
            self.logger.write('epoch time: %0.3f \n' % endtime)

            # ################# Eval ##################
            self.logger.write('Begin evaluation...\n')
            self._clear_cache()
            self.is_training = False
            self.net_G.eval()

            # Iterate over data.
            for self.batch_id, batch in enumerate(self.dataloaders['val'], 0):
                with torch.no_grad():
                    self._forward_pass(batch)
                    gt = self.batch['L'].to(self.device).long()
                    # gt = torch.cat([1 - gt, gt], dim=1).to(self.device).float()
                    self.G_loss = self._pxl_loss(self.G_pred, gt)
                    val_epoch_loss += self.G_loss.item()
                self._collect_running_batch_states()

            average_val_loss = val_epoch_loss / len(self.dataloaders['val'])
            self._collect_epoch_states(average_val_loss)

            # ########## Update_Checkpoints ###########
            self._update_val_curve()
            self._update_checkpoints()
