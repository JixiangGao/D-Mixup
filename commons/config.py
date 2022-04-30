import argparse

def get_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="train", choices=['train', 'validation', 'test'], type=str)
    parser.add_argument("--mix_style", type=str,
        choices=['mixup-all-all', 'mixup-tail-all', 'mixup-tail-tail', 'mixup-all-all-expand',
                'mixup-tail-all-expand', 'mixup-tail-tail-expand', 'no-mixup-all-all'])
    parser.add_argument('--not_use_dynamic_epoches', type=int)
    parser.add_argument('--dynamic_mode', type=str)
    parser.add_argument("--eval_mode", type=str)
    parser.add_argument('--labels_mixup_style', default='lambda', choices=['1', 'lambda'], type=str)
    parser.add_argument("--tail_threshold", type=int)
    parser.add_argument("--head_threshold", type=int)
    parser.add_argument("--tail_expand_num", type=int, default=None)
    parser.add_argument("--epoch", default=100, type=int)
    parser.add_argument("--backbone_net", type=str)
    parser.add_argument("--batch_size", default=50, type = int)
    parser.add_argument("--saved_model_dir", type=str)
    parser.add_argument("--load_model_num", default=-1)
    parser.add_argument("--label_num", default=383, type=int)
    parser.add_argument("--lr", default=1e-4, type = float)
    parser.add_argument("--lr_gamma", default=0.8, type = float)
    parser.add_argument("--step_size", default=8000, type = int)
    parser.add_argument("--input_size", default=224, choices = [224, 448], type = int)
    parser.add_argument("--inter_dim", default=2048, type = int)
    parser.add_argument("--print_freq", default=100, type=int)
    parser.add_argument("--save_freq", default=1, type=int)
    parser.add_argument("--dataset_dir", default="../../dataset/VireoFood251/", type=str)
    parser.add_argument("--use_cuda", default=True, type=bool)
    parser.add_argument("--gpu_device", default=0, )

    parser.add_argument("--use_writer", dest='use_writer', action='store_true')
    parser.add_argument("--no_writer", dest='use_writer', action='store_false')
    parser.set_defaults(use_writer=False)

    parser.add_argument("--dropout", dest='dropout', action='store_true')
    parser.add_argument("--not_dropout", dest='dropout', action='store_false')
    parser.set_defaults(dropout=False)

    parser.add_argument("--pure_dynamic", dest='pure_dynamic', action='store_true')
    parser.set_defaults(pure_dynamic=False)
    
    return parser.parse_args(argv)