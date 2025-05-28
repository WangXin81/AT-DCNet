import os
import matplotlib.pyplot as plt
import numpy as np


def curve(args):
    train_mF1 = np.load("./checkpoints/{}/train_mF1.npy".format(args.project_name))
    val_mF1 = np.load("./checkpoints/{}/val_mF1.npy".format(args.project_name))
    epochs1 = range(0, 200)
    epochs2 = range(0, 200)
    plt.figure()
    plt.plot(train_mF1, 'b', label='train_mF1')
    plt.plot(val_mF1, 'r', label='val_mF1')
    plt.ylabel('mF1')
    plt.xlabel('epoch')
    plt.legend()
    imgPath = './checkpoints/{}'.format(args.project_name)
    plt.savefig(os.path.join(imgPath, "mF1.jpg"))

    train_loss = np.load("./checkpoints/{}/train_loss.npy".format(args.project_name))
    val_loss = np.load("./checkpoints/{}/val_loss.npy".format(args.project_name))
    epochs1 = range(0, 200)
    epochs2 = range(0, 200)
    plt.figure()
    plt.plot(train_loss, 'b', label='train_loss')
    plt.plot(val_loss, 'r', label='val_loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    imgPath = './checkpoints/{}'.format(args.project_name)
    plt.savefig(os.path.join(imgPath, "loss.jpg"))
