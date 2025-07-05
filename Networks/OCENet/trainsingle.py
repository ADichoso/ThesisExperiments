import os
import torch
import torch.optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.nn.functional as F
from datetime import datetime

from model.ResNet_models import Generator, FCDiscriminator
from data import get_loader, SalObjDataset
from utils import AvgMeter, setup
from params import parser
from loss import make_confidence_label, uncertainty_aware_structure_loss

print("CuDNN", torch.backends.cudnn.version())
print("TORCH", torch.__version__) 
print("CUDA", torch.version.cuda)

def train():
    args = parser.parse_args()
    setup(args.seed)
    torch.autograd.set_detect_anomaly(True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Models
    COD_Net = Generator(channel=args.gen_reduced_channel).to(device)
    COD_Net_params = COD_Net.parameters()
    COD_Net_optimiser = torch.optim.Adam(COD_Net_params, args.lr_gen, betas=[args.beta_gen, 0.999])

    OCE_Net = FCDiscriminator().to(device)
    OCE_Net_params = OCE_Net.parameters()
    OCE_Net_optimiser = torch.optim.Adam(OCE_Net_params, args.lr_dis, betas=[args.beta_dis, 0.999])

    # DataLoader
    train_dataset = SalObjDataset(args.train_image_root, args.train_gt_root, trainsize=args.trainsize)
    train_loader = get_loader(args.train_image_root, args.train_gt_root, batchsize=args.batchsize, trainsize=args.trainsize, sampler=None)
    train_step = len(train_loader)

    # Scheduler
    COD_Net_scheduler = lr_scheduler.StepLR(OCE_Net_optimiser, step_size=20, gamma=0.1)
    CE = torch.nn.BCELoss()
    size_rates = [0.50, 0.75, 1, 1.25]

    print("Ikou!")

    for epoch in range(1, args.epoch + 1):
        COD_Net_scheduler.step()
        COD_Net.train()
        OCE_Net.train()

        loss_record_COD = AvgMeter()
        loss_record_OCE = AvgMeter()

        print('Generator learning rate: {}\nDiscriminator learning rate: {}'.format(
              COD_Net_optimiser.param_groups[0]['lr'],
              OCE_Net_optimiser.param_groups[0]['lr']))

        for i, pack in enumerate(train_loader, start=1):
            for rate in size_rates:
                COD_Net_optimiser.zero_grad()
                OCE_Net_optimiser.zero_grad()

                images, gts = pack
                images = Variable(images).to(device)
                gts = Variable(gts).to(device)

                trainsize = int(round(args.trainsize * rate / 32) * 32)
                if rate != 1:
                    images = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                    gts = F.interpolate(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

                init_pred, ref_pred = COD_Net(images)

                post_init = torch.sigmoid(init_pred.detach())
                post_ref = torch.sigmoid(ref_pred.detach())

                confi_init_label = make_confidence_label(gts=gts, pred=post_init)
                confi_ref_label = make_confidence_label(gts=gts, pred=post_ref)

                post_init = torch.cat((post_init, images), dim=1)
                post_ref = torch.cat((post_ref, images), dim=1)

                confi_init_pred = OCE_Net(post_init)
                confi_ref_pred = OCE_Net(post_ref)

                confi_loss_pred_init = CE(torch.sigmoid(confi_init_pred), confi_init_label)
                confi_loss_pred_ref = CE(torch.sigmoid(confi_ref_pred), confi_ref_label)
                OCE_loss = 0.5 * (confi_loss_pred_init + confi_loss_pred_ref)

                OCE_loss.backward()
                OCE_Net_optimiser.step()

                struct_loss1 = uncertainty_aware_structure_loss(pred=init_pred, mask=gts, confi_map=confi_init_pred.detach(), epoch=epoch)
                struct_loss2 = uncertainty_aware_structure_loss(pred=ref_pred, mask=gts, confi_map=confi_ref_pred.detach(), epoch=epoch)
                COD_loss = 0.5 * (struct_loss1 + struct_loss2)

                COD_loss.backward()
                COD_Net_optimiser.step()

                if rate == 1:
                    loss_record_COD.update(COD_loss.data, args.batchsize)
                    loss_record_OCE.update(OCE_loss.data, args.batchsize)

            if i % 10 == 0 or i == train_step:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], COD Loss: {:.4f}, OCE Loss: {:.4f}'.
                       format(datetime.now(), epoch, args.epoch, i, train_step, loss_record_COD.show(), loss_record_OCE.show()))

        if not os.path.exists(args.model_save_path):
            os.makedirs(args.model_save_path)

        if epoch % 10 == 0:
            torch.save(COD_Net.state_dict(), os.path.join(args.model_save_path, args.experiment_name, f'/COD_Model_{epoch}.pth'))
            torch.save(OCE_Net.state_dict(), os.path.join(args.model_save_path, args.experiment_name, f'/OCE_Model_{epoch}.pth'))
            print("Successfully saved the trained models to {}".format(args.model_save_path))

if __name__ == "__main__":
    train()
