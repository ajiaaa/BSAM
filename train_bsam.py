import argparse
import torch
from model.wide_res_net import WideResNet
from model.smooth_cross_entropy import smooth_crossentropy
from data.cifar import Cifar10, Cifar100
from utility.log import Log
from utility.initialize import initialize
from utility.bypass_bn import enable_running_stats, disable_running_stats
from utility.scheduler import CosineScheduler, ProportionScheduler
import sys

sys.path.append("..")
from bsam import BSAM
from model.resnet import ResNet18 as resnet18
from model.PyramidNet import PyramidNet as PYRM
from model.mobilenetv2 import MobileNetV2
import time
import random
from utility.time_record import TIME_RECORD
from utility.save_file import write_to_file, copy_files_to_folders, sivefile_config
from tqdm import tqdm
import torch.nn.functional as F
import torchvision

import matplotlib

matplotlib.use("Agg")




def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptive", default=False, type=bool, help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--batch_size", default=128, type=int,
                        help="Batch size used in the training and validation loop.")
    parser.add_argument("--depth", default=28, type=int, help="Number of layers.")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
    parser.add_argument("--epochs", default=200, type=int, help="Total number of epochs.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
    parser.add_argument("--learning_rate", default=0.05, type=float, help="Base learning rate at the start of the training.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--threads", default=0, type=int, help="Number5687 of CPU threads for dataloaders.")
    parser.add_argument("--rho_max_sharp", default=0.05, type=int, help="Rho parameter for SAM.")
    parser.add_argument("--rho_min_sharp_max", default=0.05, type=int, help="")
    parser.add_argument("--rho_min_sharp_min", default=0, type=int, help="")
    parser.add_argument("--weight_decay", default=0.001, type=float, help="L2 weight decay.")
    parser.add_argument("--width_factor", default=10, type=int, help="How many times wider compared to normal ResNet.")
    parser.add_argument("--model", default='resnet18', type=str, help="resnet18, wideresnet, pyramidnet")
    parser.add_argument("--dataset", default='cifar10', type=str, help="cifar10, cifar100")
    args = parser.parse_args()

    save_file_list, save_file_dir = sivefile_config("results/", args.dataset, args.model, "mmsam")

    index_num = random.randint(1, 2000)
    print('Seed:', index_num)
    write_to_file("other/whole_train_time.txt", 'Seed:' + str(index_num))
    initialize(args, seed=index_num)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # write_to_file("other/whole_train_time.txt", 'rho_min_sharp_max:' + str(j))

    if args.dataset == 'cifar100':
        class_num = 100
        dataset = Cifar100(args.batch_size, args.threads)
    elif args.dataset == 'cifar10':
        class_num = 10
        dataset = Cifar10(args.batch_size, args.threads)

    log = Log(log_each=10)

    model = {
        'resnet18': resnet18(num_classes=class_num).to(device),
        'wideresnet': WideResNet(28, 10, args.dropout, in_channels=3, labels=class_num).to(device),
        'pyramidnet': PYRM('cifar' + str(class_num), 110, 270, class_num, False).to(device),
        'vgg16_bn': torchvision.models.vgg16_bn(num_classes=class_num).to(device),
        'efficientnet_b0': torchvision.models.efficientnet_b0(num_classes=class_num).to(device),
        'MobileNetV2':MobileNetV2(num_classes=class_num).to(device)
    }[args.model]

    base_optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = CosineScheduler(T_max=args.epochs * len(dataset.train), max_value=args.learning_rate, min_value=0.0, optimizer=base_optimizer)
    rho_scheduler = ProportionScheduler(pytorch_lr_scheduler=scheduler, max_lr=args.learning_rate, min_lr=0.0,
                                        max_value=args.rho_min_sharp_max, min_value=args.rho_min_sharp_min)

    optimizer = BSAM(model.parameters(), base_optimizer, rho_max_sharp=args.rho_max_sharp, rho_min_sharp=args.rho_min_sharp_max, rho_scheduler=rho_scheduler,
                            adaptive=args.adaptive, lr=args.learning_rate)

    whole_time = 0
    best_result = 0

    for epoch in range(args.epochs):
        model.train()
        start_time = time.time()

        for batch in tqdm(dataset.train):
            inputs, targets = batch[0].to(device), batch[1].to(device)

            enable_running_stats(model)
            predictions = model(inputs)
            loss = smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing)
            loss.mean().backward()
            optimizer.to_max_point(zero_grad=True)

            disable_running_stats(model)
            predictions = model(inputs)
            lossa = smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing)
            lossa.mean().backward()
            optimizer.to_min_point(zero_grad=True)

            predictions = model(inputs)
            lossb = smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing)
            lossb.mean().backward()
            optimizer.opt_max_min_step(epoch, zero_grad=True)

            with torch.no_grad():
                lr_ = scheduler.step()
                optimizer.update_rho_t()

        end_time = time.time()
        es_time = end_time - start_time
        whole_time += es_time
        write_to_file("other/whole_train_time.txt", str(whole_time))

        model.eval()
        log.eval(len_dataset=len(dataset.test))

        with torch.no_grad():
            for batch in dataset.test:
                inputs, targets = batch[0].to(device), batch[1].to(device)  # (b.to(device) for b in batch)

                predictions = model(inputs)
                loss = smooth_crossentropy(predictions, targets)
                correct = torch.argmax(predictions, 1) == targets
                log(model, loss.cpu(), correct.cpu())
            log.flush()
            write_to_file("other/accuracy.txt", str(log.acc))

        # if log.acc > best_result:
        #     torch.save(model, save_file_dir + 'best.pth')
        #     best_result = log.acc


        #torch.save(model.state_dict(), save_file_dir + str(epoch)+'_model_sam.pkl')

    copy_files_to_folders(save_file_list, save_file_dir)

if __name__ == "__main__":
    train()




