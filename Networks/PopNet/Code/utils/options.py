import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--epoch',       type=int,   default=100,   help='epoch number')
parser.add_argument('--lr',          type=float, default=1e-4,  help='learning rate')
parser.add_argument('--batchsize',   type=int,   default=4,    help='training batch size')
parser.add_argument('--trainsize',   type=int,   default=352,   help='training dataset size')
parser.add_argument('--clip',        type=float, default=0.5,   help='gradient clipping margin')
parser.add_argument('--lw',          type=float, default=0.001, help='weight')
parser.add_argument('--decay_rate',  type=float, default=0.1,   help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int,   default=60,    help='every n epochs decay learning rate')
#parser.add_argument('--load',        type=str,   default='./SPNet_epoch_best.pth',  help='train from checkpoints') # if continue learning from previous stage
parser.add_argument('--load',        type=str,   default='',  help='train from checkpoints')

parser.add_argument('--gpu_id',      type=str,   default='0',   help='train use gpu')
parser.add_argument('--dataset',      type=str,   default='ACOD-12K',   help='Dataset')

parser.add_argument('--rgb_label_root',      type=str, default='./Datasets/',           help='the training rgb images root')
parser.add_argument('--depth_label_root',    type=str, default='./Datasets/',         help='the training depth images root')
parser.add_argument('--gt_label_root',       type=str, default='./Datasets/',            help='the training gt images root')

# or use other datasets for validation
parser.add_argument('--val_rgb_root',        type=str, default='./Datasets/',      help='the test rgb images root')
parser.add_argument('--val_depth_root',      type=str, default='./Datasets/',    help='the test depth images root')
parser.add_argument('--val_gt_root',         type=str, default='./Datasets/',       help='the test gt images root')

parser.add_argument('--save_path',           type=str, default='./Checkpoints/PopNet/',    help='the path to save models and logs')

opt = parser.parse_args()

opt.rgb_label_root = opt.rgb_label_root + opt.dataset + '/Train/Imgs/'
opt.depth_label_root = opt.depth_label_root + opt.dataset + '/Train/PopNet_Depth/'
opt.gt_label_root = opt.gt_label_root + opt.dataset + '/Train/GT/'
opt.val_rgb_root = opt.val_rgb_root + opt.dataset + '/Test/Imgs/'
opt.val_depth_root = opt.val_depth_root + opt.dataset + '/Test/PopNet_Depth/'
opt.val_gt_root = opt.val_gt_root + opt.dataset + '/Test/GT/'
opt.save_path = opt.save_path + opt.dataset + '/'
