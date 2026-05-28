import argparse
from inferenceADC import inferenceADC
from inferenceCRF import inferenceCRF
from src.ADCRezero.test.evaluation import evaluation
import os
import datetime

def test():
    parser = argparse.ArgumentParser(description="Batched inference for CReZero family models")
    parser.add_argument('--inference', action='store_true', help="Run inference")
    parser.add_argument('--evaluation', action='store_true', help="Run evaluation")
    parser.add_argument('--ARezero_path', type=str, help="Path to the pretrained ARezero model")
    parser.add_argument('--DRezero_path', type=str, help="Path to the pretrained DRezero model")
    parser.add_argument('--CRezero_path', type=str, help="Path to the pretrained CRezero model")
    parser.add_argument('--test_dir', required=True, help="Directory of test data")
    parser.add_argument('--output_dir', default='outputs', help="Directory to save outputs")
    parser.add_argument('--config', default='src/ADCRezero/config/original.yaml', help="Path to config file")
    parser.add_argument('--device', default='cuda', help="Device to use for inference")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for inference")
    parser.add_argument('--num_workers', type=int, default=16, help="Number of workers for data loading")
    parser.add_argument('--mic_arch', choices=['circular', 'linear'], default='circular', help="Microphone array architecture")
    parser.add_argument('--region_type', choices=['angular','spherical','conical', 'ring', 'fan'], default='conical', help="Type of query region")
    parser.add_argument('--inferenceCRF', action='store_true', help="Use CRF for inference")
    parser.add_argument('--fan_method', choices=['AR', 'AD', 'CC'], default='CC', help="Method for fan region: AR (Angular+Ring), AD (Angular+Spherical), CC (Conical-Conical)")
    args = parser.parse_args()
    
    now = str(datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))
    if args.region_type == 'angular':
        args.est_dir = os.path.join(args.output_dir, args.ARezero_path.split("/")[-3], args.ARezero_path.split("/")[-2], args.ARezero_path.split("/")[-1].split(".")[0])
    elif args.region_type == 'spherical':
        args.est_dir = os.path.join(args.output_dir, args.DRezero_path.split("/")[-3], args.DRezero_path.split("/")[-2], args.DRezero_path.split("/")[-1].split(".")[0])
    elif args.region_type == 'conical':
        args.est_dir = os.path.join(args.output_dir, args.CRezero_path.split("/")[-3], args.CRezero_path.split("/")[-2], args.CRezero_path.split("/")[-1].split(".")[0])
    elif args.region_type == 'ring' or args.region_type == 'fan':
        args.est_dir = os.path.join(args.output_dir, args.region_type, now)
    else:
        raise ValueError("Unknown region type. args.region_type must be angular, spherical, conical, ring or fan.")
    
    if args.inference:
        print("===========Inference Start===========")
        if not args.inferenceCRF:
            inferenceADC(args)
        else:
            inferenceCRF(args)
    else:
        print("Skip Inference")
    
    if args.evaluation:
        print("===========Evaluation Start===========")
        evaluation(args)
    else:
        print("Skip Evaluation")

if __name__ == '__main__':
    test()