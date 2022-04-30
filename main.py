from run_network import RunNetwork
from commons.config import get_args
import sys

"""
Run command like the followings.

python main.py --mode train --gpu_device 0 --saved_model_dir \
LongTail-383-mixup-all-20-100 --print_freq 100 --load_model_num -1 \
--tail_threshold 20 --head_threshold 100  --no_writer 

python main.py --mode train --gpu_device 1 --saved_model_dir \
LongTail-383-mixup-all-100-1000 --print_freq 100 --load_model_num -1 \
--tail_threshold 100 --head_threshold 1000  --no_writer 

"""

if __name__ == '__main__':
    args = get_args(sys.argv[1:])
    running = RunNetwork(args)

    if args.mode == 'train':
        running.train()
    else:
        p, r, micro_F1, macro_F1, F1_sum = running.eval() 
        print("epoch {} | P {:.4f} | R {:.4f} | micro_F1 {:.4f} | macro_F1 {:.4f}".format(
            args.load_model_num, p, r, micro_F1, macro_F1))
        for i in F1_sum:
            print(i.numpy())

        if args.use_writer:
            writer = SummaryWriter(comment="-%s-%s"%(args.mode, args.saved_model_dir))
            writer.add_text('Eval', running._get_format_args())
            writer.add_scalars('LongTail-%d/Performance' % args.label_num, 
                {'Precision':p, 'Recall':r, 'micro_F1': micro_F1,'macro_F1': macro_F1}, 
                i+1)

