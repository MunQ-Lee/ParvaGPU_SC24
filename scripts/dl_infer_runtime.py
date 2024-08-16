import os
import argparse
import time
import atexit

import torch
from torchvision.models import densenet121
from torchvision.models import densenet169
from torchvision.models import densenet201
from torchvision.models import inception_v3
from torchvision.models import mobilenet_v2
from torchvision.models import resnet101
from torchvision.models import resnet152
from torchvision.models import resnet50
from torchvision.models import vgg16
from torchvision.models import vgg19
import bert.modeling as modeling

# arument setting
parser = argparse.ArgumentParser(description='deep learning workload execution')
parser.add_argument("-m", "--model", default='resnet50', help="select model")
parser.add_argument('-g', "--gpus", default='0', type=str, help="select one gpu")
parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N',
                    help='mini-batch size (default: 64), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-s', '--steps', default=0, type=int, metavar='N', help='number of steps')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]=args.gpus

num_images=0
total_time=0
num_steps=0

# main function
def main():
    print(f"{args.model}-{args.batch_size} Model building...")
    model = build(model=args.model)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
        
    model.to(device)
    model.eval()
    
    global num_images
    global total_time
    global num_steps
    init_iter = 10
    # atexit.register(handle_exit_tc)

    if args.model == "bert":
        sequence_length = 128

        input_ids_cpu = torch.ones(args.batch_size, sequence_length, dtype=torch.long)
        input_mask_cpu = torch.ones(args.batch_size, sequence_length, dtype=torch.long)
        segment_ids_cpu = torch.ones(args.batch_size, sequence_length, dtype=torch.long)
        for  _ in range(init_iter):
            input_ids = input_ids_cpu.to(device)
            input_mask = input_mask_cpu.to(device)
            segment_ids = segment_ids_cpu.to(device)
            output = model(input_ids, segment_ids, input_mask)

    else:

        images_cpu = torch.ones(args.batch_size, 3, 224, 224)
        for  _ in range(init_iter):
            images_gpu = images_cpu.to(device)
            output = model(images_gpu)

        
    print("Start inference...")
    i = 0

    while True:
        i += 1

        start_time = time.time()
        if args.model == "bert":
            input_ids = input_ids_cpu.to(device)
            input_mask = input_mask_cpu.to(device)
            segment_ids = segment_ids_cpu.to(device)
            output = model(input_ids, segment_ids, input_mask)
        else:
            images_gpu = images_cpu.to(device)
            output = model(images_gpu)
        finish_time = time.time()

        step_time = finish_time - start_time
        total_time += step_time
        num_images += args.batch_size
        num_steps += 1

        print(f'\r inference: {i:5d} | throughput: {args.batch_size/step_time:7.3f} | time: {step_time:7.3f}', end='')
        # if args.steps != 0:
        #     print(f'\r inference: {i:5d} | throughput: {args.batch_size/step_time:7.3f} | time: {step_time:7.3f}', end='')
        if i == args.steps:
            break


# Model build
def build(model='resnet50'):

    if model=='bert':
        num_label = 2
        config = modeling.BertConfig.from_json_file("./bert/large.json")
        if config.vocab_size % 8 != 0:
            config.vocab_size += 8 - (config.vocab_size % 8)
        buildingModel = modeling.BertForSequenceClassification(config, num_labels=num_label)

    elif model=='densenet121':
        buildingModel = densenet121(pretrained=None)
    elif model=='densenet169':
        buildingModel = densenet169(pretrained=None)
    elif model=='densenet201':
        buildingModel = densenet201(pretrained=None)
    elif model=='inceptionv3':
        buildingModel = inception_v3(pretrained=None)
    elif model=='mobilenetv2':
        buildingModel = mobilenet_v2(pretrained=None)
    elif model=='resnet50':
        buildingModel = resnet50(pretrained=None)
    elif model=='resnet101':
        buildingModel = resnet101(pretrained=None)
    elif model=='resnet152':
        buildingModel = resnet152(pretrained=None)
    elif model=='vgg16':
        buildingModel = vgg16(pretrained=None)
    elif model=='vgg19':
        buildingModel = vgg19(pretrained=None)
    else : 
        print("Unknown model!")
        return None

    return buildingModel

def handle_exit_tc():
    global num_images
    global total_time
    global num_steps

    print(f'\n\nTotal inference throughput: {num_images/total_time:7.5f}, Total inference latency: {total_time/num_steps:7.5f}')

if __name__ == "__main__":
    main()


