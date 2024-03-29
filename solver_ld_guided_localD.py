from model import Generator
from model import Discriminator
from model import localD
from torch.autograd import Variable
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
from landmark2heatmap import process_oneimg, detect_landmark
from landmark2heatmap_onlyeyemouth import process_oneimg_, detect_landmark_
import cv2
from random import randint
from skimage import io
from skimage import color
from torchvision import transforms as T
from PIL import Image
from tensorboardX import SummaryWriter
summary = SummaryWriter()


class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, celeba_loader, rafd_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.celeba_loader = celeba_loader
        self.rafd_loader = rafd_loader

        # Model configurations.
        self.c_dim = config.c_dim
        self.c2_dim = config.c2_dim
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp
        self.lambda_rec_ld = config.lambda_rec_ld  # added
        self.lambda_fake_ld = config.lambda_fake_ld  # added
        self.locald_conv_dim = config.locald_conv_dim   # added for ld_guided_localD
        self.locald_repeat_num = config.locald_repeat_num  # added for ld_guided_localD
        self.lambda_local = config.lambda_local  # added for ld_guided_localD

        # Training configurations.
        self.dataset = config.dataset
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.locald_lr = config.locald_lr  # added for ld_guided_localD
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters
        self.selected_attrs = config.selected_attrs

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        """Create a generator and a discriminator."""
        if self.dataset in ['CelebA', 'RaFD']:
            self.G = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num)
            self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num)
            self.localD = localD(self.d_conv_dim, self.locald_repeat_num)  # added
        elif self.dataset in ['Both']:
            self.G = Generator(self.g_conv_dim, self.c_dim + self.c2_dim + 2, self.g_repeat_num)  # 2 for mask vector.
            self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim + self.c2_dim, self.d_repeat_num)

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.locald_optimizer = torch.optim.Adam(self.localD.parameters(), self.locald_lr, [self.beta1, self.beta2])  # added
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')
        self.print_network(self.localD, 'localD')  # added

        self.G.to(self.device)
        self.D.to(self.device)
        self.localD.to(self.device)  # added

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        localD_path = os.path.join(self.model_save_dir, '{}-localD.ckpt'.format(resume_iters))  # added
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))
        self.localD.load_state_dict(torch.load(localD_path, map_location=lambda storage, loc: storage))  # added

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr, d_lr, locald_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr
        for param_group in self.locald_optimizer.param_groups:  # added
            param_group['lr'] = locald_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()
        self.locald_optimizer.zero_grad()  # added

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
        return torch.mean((dydx_l2norm - 1) ** 2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def create_labels(self, c_org, c_dim=5, dataset='CelebA', selected_attrs=None):
        """Generate target domain labels for debugging and testing."""
        # Get hair color indices.
        if dataset == 'CelebA':
            hair_color_indices = []
            for i, attr_name in enumerate(selected_attrs):
                if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                    hair_color_indices.append(i)

        c_trg_list = []
        for i in range(c_dim):
            if dataset == 'CelebA':
                c_trg = c_org.clone()
                if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
                    c_trg[:, i] = 1
                    for j in hair_color_indices:
                        if j != i:
                            c_trg[:, j] = 0
                else:
                    c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.
            elif dataset == 'RaFD':
                c_trg = self.label2onehot(torch.ones(c_org.size(0)) * i, c_dim)

            c_trg_list.append(c_trg.to(self.device))
        return c_trg_list

    def classification_loss(self, logit, target, dataset='CelebA'):
        """Compute binary or softmax cross entropy loss."""
        if dataset == 'CelebA':
            return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)
        elif dataset == 'RaFD':
            return F.cross_entropy(logit, target)

 # --------------------------------------------------------------------------------------------------------------------
    def train(self):
        """Train StarGAN within a single dataset."""
        # Set data loader.
        if self.dataset == 'CelebA':
            data_loader = self.celeba_loader
        elif self.dataset == 'RaFD':
            data_loader = self.rafd_loader
        # print(len(data_loader.dataset.imgs))

        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)
        x_fixed, c_org, fp = next(data_iter)  # added
        x_fixed = x_fixed.to(self.device)
        c_fixed_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)


        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr
        locald_lr = self.locald_lr  # added

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real images and labels.
            try:
                x_real, label_org, fp = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real, label_org, fp = next(data_iter)

            # print(os.path.basename(list(fp)[1]))


            # Generate target domain labels randomly.
            rand_idx = torch.randperm(label_org.size(0))
            label_trg = label_org[rand_idx]

            if self.dataset == 'CelebA':
                c_org = label_org.clone()
                c_trg = label_trg.clone()
            elif self.dataset == 'RaFD':
                c_org = self.label2onehot(label_org, self.c_dim)
                c_trg = self.label2onehot(label_trg, self.c_dim)

            # print(c_trg)

            for j in range(16):
                rand_num = os.path.basename(list(fp)[j])
                if c_trg[j][0] == 1:
                    c_trg_ld = process_oneimg('D:\\Workspace\\0. StarGAN\\data\\RaFD\\train\\angry\\', rand_num)
                elif c_trg[j][1] == 1:
                    c_trg_ld = process_oneimg('D:\\Workspace\\0. StarGAN\\data\\RaFD\\train\\contemptuous\\', rand_num)
                elif c_trg[j][2] == 1:
                    c_trg_ld = process_oneimg('D:\\Workspace\\0. StarGAN\\data\\RaFD\\train\\disgusted\\', rand_num)
                elif c_trg[j][3] == 1:
                    c_trg_ld = process_oneimg('D:\\Workspace\\0. StarGAN\\data\\RaFD\\train\\fearful\\', rand_num)
                elif c_trg[j][4] == 1:
                    c_trg_ld = process_oneimg('D:\\Workspace\\0. StarGAN\\data\\RaFD\\train\\happy\\', rand_num)
                elif c_trg[j][5] == 1:
                    c_trg_ld = process_oneimg('D:\\Workspace\\0. StarGAN\\data\\RaFD\\train\\neutral\\', rand_num)
                elif c_trg[j][6] == 1:
                    c_trg_ld = process_oneimg('D:\\Workspace\\0. StarGAN\\data\\RaFD\\train\\sad\\', rand_num)
                elif c_trg[j][7] == 1:
                    c_trg_ld = process_oneimg('D:\\Workspace\\0. StarGAN\\data\\RaFD\\train\\surprised\\', rand_num)
                cv2.imwrite('D:\\Workspace\\0. StarGAN\\heatmap_c_trg_sample.jpg',c_trg_ld)
                c_trg_ld = torch.Tensor(cv2.imread('heatmap_c_trg_sample.jpg'))


            x_real = x_real.to(self.device)  # Input images.
            c_org = c_org.to(self.device)  # Original domain labels.
            c_trg = c_trg.to(self.device)  # Target domain labels.
            c_trg_ld = c_trg_ld.to(self.device)  # added

            label_org = label_org.to(self.device)  # Labels for computing classification loss.
            label_trg = label_trg.to(self.device)  # Labels for computing classification loss.

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # Compute loss with real images.
            out_src, out_cls = self.D(x_real)
            d_loss_real = - torch.mean(out_src)
            d_loss_cls = self.classification_loss(out_cls, label_org, self.dataset)

            # Compute loss with fake images.
            # x_fake = self.G(x_real, c_trg)
            x_fake = self.G(x_real, c_trg, c_trg_ld)  # added
            out_src, out_cls = self.D(x_fake.detach())
            d_loss_fake = torch.mean(out_src)

            # added - local discriminator
            for j in range(16):
                save_image(x_fake[j], 'd_fake_sample.jpg')
                lm = detect_landmark('D:\\Workspace\\0. StarGAN\\','d_fake_sample.jpg')

                if lm is None:
                    mouth_start_x = 0
                    mouth_finish_x = 50
                    mouth_start_y = 0
                    mouth_finish_y = 50

                    lefteye_start_x = 0
                    lefteye_finish_x = 50
                    lefteye_start_y = 0
                    lefteye_finish_y = 50

                    righteye_start_x =0
                    righteye_finish_x = 50
                    righteye_start_y = 0
                    righteye_finish_y = 50
                else:
                    mouth_start_x = lm[49][0] - 10
                    mouth_finish_x = lm[55][0] + 10
                    mouth_start_y = max(lm[51][1], lm[52][1], lm[53][1]) - 10
                    mouth_finish_y = lm[58][1] + 15

                    lefteye_start_x = lm[37][0] - 15
                    lefteye_finish_x = lm[40][0] + 15
                    lefteye_start_y = min(lm[41][1], lm[42][1]) - 20
                    lefteye_finish_y = max(lm[38][1], lm[39][1]) + 15

                    righteye_start_x = lm[43][0] - 15
                    righteye_finish_x = lm[46][0] + 15
                    righteye_start_y = min(lm[48][1], lm[47][1]) - 20
                    righteye_finish_y = max(lm[44][1], lm[45][1]) + 15

                lm_image = cv2.imread('D:\\Workspace\\0. StarGAN\\d_fake_sample.jpg')
                cv2.imwrite('D:\\Workspace\\0. StarGAN\\d_fake_sample_mouth.jpg',lm_image[mouth_start_y:mouth_finish_y,mouth_start_x:mouth_finish_x])
                cv2.imwrite('D:\\Workspace\\0. StarGAN\\d_fake_sample_lefteye.jpg',lm_image[lefteye_start_y:lefteye_finish_y,lefteye_start_x:lefteye_finish_x])
                cv2.imwrite('D:\\Workspace\\0. StarGAN\\d_fake_sample_righteye.jpg',lm_image[righteye_start_y:righteye_finish_y,righteye_start_x:righteye_finish_x])

                im_lefteye = cv2.imread('D:\\Workspace\\0. StarGAN\\d_fake_sample_lefteye.jpg')
                im_righteye = cv2.imread('D:\\Workspace\\0. StarGAN\\d_fake_sample_righteye.jpg')
                im_mouth = cv2.imread('D:\\Workspace\\0. StarGAN\\d_fake_sample_mouth.jpg')

                im_lefteye = T.ToTensor()(im_lefteye)
                im_lefteye = T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(im_lefteye)
                im_lefteye = im_lefteye.unsqueeze(0)

                im_righteye = T.ToTensor()(im_righteye)
                im_righteye = T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(im_righteye)
                im_righteye = im_righteye.unsqueeze(0)

                im_mouth = T.ToTensor()(im_mouth)
                im_mouth = T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(im_mouth)
                im_mouth = im_mouth.unsqueeze(0)

                # x_fake = x_fake.to(self.device)
                im_lefteye= im_lefteye.to(self.device)
                im_righteye = im_righteye.to(self.device)
                im_mouth = im_mouth.to(self.device)
                d_loss_eyes = torch.mean(self.localD(im_lefteye.detach())) + torch.mean(self.localD(im_righteye.detach()))
                d_loss_mouth = torch.mean(self.localD(im_mouth.detach()))
                d_loss_local = d_loss_eyes + d_loss_mouth

            # Compute loss for gradient penalty.
            alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
            out_src, _ = self.D(x_hat)
            d_loss_gp = self.gradient_penalty(out_src, x_hat)

            # Backward and optimize.
            # d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp
            d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp + self.lambda_local * d_loss_local
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Logging.
            loss = {}
            loss['D/loss_real'] = d_loss_real.item()
            loss['D/loss_fake'] = d_loss_fake.item()
            loss['D/loss_cls'] = d_loss_cls.item()
            loss['D/loss_gp'] = d_loss_gp.item()
            loss['D/loss_local'] = d_loss_local.item()  # added

            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #

            if (i + 1) % self.n_critic == 0:
                # Original-to-target domain.
                # x_fake = self.G(x_real, c_trg)
                x_fake = self.G(x_real, c_trg, c_trg_ld)  # added
                out_src, out_cls = self.D(x_fake)
                g_loss_fake = - torch.mean(out_src)
                g_loss_cls = self.classification_loss(out_cls, label_trg, self.dataset)

                # Target-to-original domain.
                # x_reconst = self.G(x_fake, c_org)
                for j in range(16):
                    save_image(x_real[j], 'real_sample.jpg')
                    cv2.imwrite('D:\\Workspace\\0. StarGAN\\heatmap_real_sample.jpg', process_oneimg('D:\\Workspace\\0. StarGAN\\', 'real_sample.jpg'))
                    x_real_ld = torch.Tensor(cv2.imread('heatmap_real_sample.jpg'))

                x_real_ld = x_real_ld.to(self.device)  # added
                x_reconst = self.G(x_fake, c_org, x_real_ld)  # added
                g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                # added
                for j in range(16):
                    save_image(x_fake[j], 'fake_sample.jpg')
                    cv2.imwrite('D:\\Workspace\\0. StarGAN\\heatmap_fake_sample.jpg', process_oneimg('D:\\Workspace\\0. StarGAN\\',  'fake_sample.jpg'))
                    x_fake_ld = torch.Tensor(cv2.imread('heatmap_fake_sample.jpg'))

                    save_image(x_reconst[j], 'reconst_sample.jpg')
                    cv2.imwrite('D:\\Workspace\\0. StarGAN\\heatmap_reconst_sample.jpg', process_oneimg('D:\\Workspace\\0. StarGAN\\', 'reconst_sample.jpg'))
                    x_reconst_ld = torch.Tensor(cv2.imread('heatmap_reconst_sample.jpg'))

                x_real_ld = x_real_ld.to(self.device)  # added
                x_reconst_ld = x_reconst_ld.to(self.device) # added
                x_fake_ld = x_fake_ld.to(self.device) # added
                # Backward and optimize.
                g_loss_rec_ld = torch.nn.MSELoss()(x_real_ld, x_reconst_ld) / (128*128) # added
                g_loss_fake_ld = torch.nn.MSELoss()(c_trg_ld, x_fake_ld)  / (128*128) # added
                # g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls
                g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls + g_loss_rec_ld * self.lambda_rec_ld + g_loss_fake_ld * self.lambda_fake_ld  # added

                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_rec'] = g_loss_rec.item()
                loss['G/loss_cls'] = g_loss_cls.item()
                loss['G/loss_rec_ld'] = g_loss_rec_ld.item()  # added
                loss['G/loss_fake_ld'] = g_loss_fake_ld.item()  # added

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i + 1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i + 1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i + 1)

            # Translate fixed images for debugging.
            if (i + 1) % self.sample_step == 0:
                with torch.no_grad():
                    x_fake_list = [x_fixed]
                    for c_fixed in c_fixed_list:
                        x_fake_list.append(self.G(x_fixed, c_fixed, x_real_ld))  # added
                    x_concat = torch.cat(x_fake_list, dim=3)
                    sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i + 1))
                    save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(sample_path))

            # Save model checkpoints.
            if (i + 1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i + 1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i + 1))
                localD_path = os.path.join(self.model_save_dir, '{}-localD.ckpt'.format(i + 1))  # added
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                torch.save(self.localD.state_dict(), localD_path)  # added
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i + 1) % self.lr_update_step == 0 and (i + 1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                locald_lr -= (self.locald_lr / float(self.num_iters_decay))  # added
                self.update_lr(g_lr, d_lr, self.locald_lr)  # added
                print('Decayed learning rates, g_lr: {}, d_lr: {}, locald_lr:{}.'.format(g_lr, d_lr, locald_lr))  # added

            # added for loss histoty
            if (i + 1) % 1000 == 0:
                summary.add_scalar('D/loss_real', d_loss_real.item(), i+1)
                summary.add_scalar('D/loss_fake', d_loss_fake.item(), i+1)
                summary.add_scalar('D/loss_gp', d_loss_gp.item(), i + 1)
                summary.add_scalar('D/loss_local', d_loss_local.item(), i+1)
                summary.add_scalar('G/loss_fake', g_loss_fake.item(), i+1)
                summary.add_scalar('G/loss_fake', g_loss_fake.item(), i + 1)
                summary.add_scalar('G/loss_rec', g_loss_rec.item(), i + 1)
                summary.add_scalar('G/loss_cls', g_loss_cls.item(), i + 1)
                summary.add_scalar('G/loss_rec_ld', g_loss_rec_ld.item(), i + 1)
                summary.add_scalar('G/loss_fake_ld', g_loss_fake_ld.item(), i + 1)
                summary.add_scalar('G/loss', g_loss.item(), i + 1)
                summary.add_scalar('D/loss', d_loss.item(), i + 1)

    # ----------------------------------------------------------------------------------------------------------------------
    def train_multi(self):
        """Train StarGAN with multiple datasets."""
        # Data iterators.
        celeba_iter = iter(self.celeba_loader)
        rafd_iter = iter(self.rafd_loader)

        # Fetch fixed inputs for debugging.
        x_fixed, c_org = next(celeba_iter)
        x_fixed = x_fixed.to(self.device)
        c_celeba_list = self.create_labels(c_org, self.c_dim, 'CelebA', self.selected_attrs)
        c_rafd_list = self.create_labels(c_org, self.c2_dim, 'RaFD')
        zero_celeba = torch.zeros(x_fixed.size(0), self.c_dim).to(self.device)  # Zero vector for CelebA.
        zero_rafd = torch.zeros(x_fixed.size(0), self.c2_dim).to(self.device)  # Zero vector for RaFD.
        mask_celeba = self.label2onehot(torch.zeros(x_fixed.size(0)), 2).to(self.device)  # Mask vector: [1, 0].
        mask_rafd = self.label2onehot(torch.ones(x_fixed.size(0)), 2).to(self.device)  # Mask vector: [0, 1].

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):
            for dataset in ['CelebA', 'RaFD']:

                # =================================================================================== #
                #                             1. Preprocess input data                                #
                # =================================================================================== #

                # Fetch real images and labels.
                data_iter = celeba_iter if dataset == 'CelebA' else rafd_iter

                try:
                    x_real, label_org = next(data_iter)
                except:
                    if dataset == 'CelebA':
                        celeba_iter = iter(self.celeba_loader)
                        x_real, label_org = next(celeba_iter)
                    elif dataset == 'RaFD':
                        rafd_iter = iter(self.rafd_loader)
                        x_real, label_org = next(rafd_iter)

                # Generate target domain labels randomly.
                rand_idx = torch.randperm(label_org.size(0))
                label_trg = label_org[rand_idx]

                if dataset == 'CelebA':
                    c_org = label_org.clone()
                    c_trg = label_trg.clone()
                    zero = torch.zeros(x_real.size(0), self.c2_dim)
                    mask = self.label2onehot(torch.zeros(x_real.size(0)), 2)
                    c_org = torch.cat([c_org, zero, mask], dim=1)
                    c_trg = torch.cat([c_trg, zero, mask], dim=1)
                elif dataset == 'RaFD':
                    c_org = self.label2onehot(label_org, self.c2_dim)
                    c_trg = self.label2onehot(label_trg, self.c2_dim)
                    zero = torch.zeros(x_real.size(0), self.c_dim)
                    mask = self.label2onehot(torch.ones(x_real.size(0)), 2)
                    c_org = torch.cat([zero, c_org, mask], dim=1)
                    c_trg = torch.cat([zero, c_trg, mask], dim=1)

                x_real = x_real.to(self.device)  # Input images.
                c_org = c_org.to(self.device)  # Original domain labels.
                c_trg = c_trg.to(self.device)  # Target domain labels.
                label_org = label_org.to(self.device)  # Labels for computing classification loss.
                label_trg = label_trg.to(self.device)  # Labels for computing classification loss.

                # =================================================================================== #
                #                             2. Train the discriminator                              #
                # =================================================================================== #

                # Compute loss with real images.
                out_src, out_cls = self.D(x_real)
                out_cls = out_cls[:, :self.c_dim] if dataset == 'CelebA' else out_cls[:, self.c_dim:]
                d_loss_real = - torch.mean(out_src)
                d_loss_cls = self.classification_loss(out_cls, label_org, dataset)

                # Compute loss with fake images.
                x_fake = self.G(x_real, c_trg)
                out_src, _ = self.D(x_fake.detach())
                d_loss_fake = torch.mean(out_src)

                # Compute loss for gradient penalty.
                alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
                x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
                out_src, _ = self.D(x_hat)
                d_loss_gp = self.gradient_penalty(out_src, x_hat)

                # Backward and optimize.
                d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp
                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # Logging.
                loss = {}
                loss['D/loss_real'] = d_loss_real.item()
                loss['D/loss_fake'] = d_loss_fake.item()
                loss['D/loss_cls'] = d_loss_cls.item()
                loss['D/loss_gp'] = d_loss_gp.item()

                # =================================================================================== #
                #                               3. Train the generator                                #
                # =================================================================================== #

                if (i + 1) % self.n_critic == 0:
                    # Original-to-target domain.
                    x_fake = self.G(x_real, c_trg)
                    out_src, out_cls = self.D(x_fake)
                    out_cls = out_cls[:, :self.c_dim] if dataset == 'CelebA' else out_cls[:, self.c_dim:]
                    g_loss_fake = - torch.mean(out_src)
                    g_loss_cls = self.classification_loss(out_cls, label_trg, dataset)

                    # Target-to-original domain.
                    x_reconst = self.G(x_fake, c_org)
                    g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                    # Backward and optimize.
                    g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls
                    self.reset_grad()
                    g_loss.backward()
                    self.g_optimizer.step()

                    # Logging.
                    loss['G/loss_fake'] = g_loss_fake.item()
                    loss['G/loss_rec'] = g_loss_rec.item()
                    loss['G/loss_cls'] = g_loss_cls.item()

                # =================================================================================== #
                #                                 4. Miscellaneous                                    #
                # =================================================================================== #

                # Print out training info.
                if (i + 1) % self.log_step == 0:
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    log = "Elapsed [{}], Iteration [{}/{}], Dataset [{}]".format(et, i + 1, self.num_iters, dataset)
                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)

                    if self.use_tensorboard:
                        for tag, value in loss.items():
                            self.logger.scalar_summary(tag, value, i + 1)

            # Translate fixed images for debugging.
            if (i + 1) % self.sample_step == 0:
                with torch.no_grad():
                    x_fake_list = [x_fixed]
                    for c_fixed in c_celeba_list:
                        c_trg = torch.cat([c_fixed, zero_rafd, mask_celeba], dim=1)
                        x_fake_list.append(self.G(x_fixed, c_trg))
                    for c_fixed in c_rafd_list:
                        c_trg = torch.cat([zero_celeba, c_fixed, mask_rafd], dim=1)
                        x_fake_list.append(self.G(x_fixed, c_trg))
                    x_concat = torch.cat(x_fake_list, dim=3)
                    sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i + 1))
                    save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(sample_path))

            # Save model checkpoints.
            if (i + 1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i + 1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i + 1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i + 1) % self.lr_update_step == 0 and (i + 1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr, self.locald_lr)
                print('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))


    def test(self):
        """Translate images using StarGAN trained on a single dataset."""
        # Load the trained generator.
        self.restore_model(self.test_iters)

        # Set data loader.
        if self.dataset == 'CelebA':
            data_loader = self.celeba_loader
        elif self.dataset == 'RaFD':
            data_loader = self.rafd_loader

        with torch.no_grad():
            for i, (x_real, c_org, fp) in enumerate(data_loader):

                # Prepare input images and target domain labels.
                x_real = x_real.to(self.device)
                c_trg_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)

                for j in range(1):
                    save_image(x_real[j], 'real_sample_test.jpg')
                    cv2.imwrite('D:\\Workspace\\0. StarGAN\\heatmap_real_sample_test.jpg', process_oneimg('D:\\Workspace\\0. StarGAN\\', 'real_sample_test.jpg'))
                    x_real_ld_test = torch.Tensor(cv2.imread('heatmap_real_sample_test.jpg'))
                x_real_ld_test = x_real_ld_test.to(self.device)  # added

                # Translate images.
                x_fake_list = [x_real]
                for c_trg in c_trg_list:
                    x_fake_list.append(self.G(x_real, c_trg, x_real_ld_test))

                # Save the translated images.
                x_concat = torch.cat(x_fake_list, dim=3)
                result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i + 1))
                save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                print('Saved real and fake images into {}...'.format(result_path))

    def test_multi(self):
        """Translate images using StarGAN trained on multiple datasets."""
        # Load the trained generator.
        self.restore_model(self.test_iters)

        with torch.no_grad():
            for i, (x_real, c_org) in enumerate(self.celeba_loader):

                # Prepare input images and target domain labels.
                x_real = x_real.to(self.device)
                c_celeba_list = self.create_labels(c_org, self.c_dim, 'CelebA', self.selected_attrs)
                c_rafd_list = self.create_labels(c_org, self.c2_dim, 'RaFD')
                zero_celeba = torch.zeros(x_real.size(0), self.c_dim).to(self.device)  # Zero vector for CelebA.
                zero_rafd = torch.zeros(x_real.size(0), self.c2_dim).to(self.device)  # Zero vector for RaFD.
                mask_celeba = self.label2onehot(torch.zeros(x_real.size(0)), 2).to(self.device)  # Mask vector: [1, 0].
                mask_rafd = self.label2onehot(torch.ones(x_real.size(0)), 2).to(self.device)  # Mask vector: [0, 1].

                # Translate images.
                x_fake_list = [x_real]
                for c_celeba in c_celeba_list:
                    c_trg = torch.cat([c_celeba, zero_rafd, mask_celeba], dim=1)
                    x_fake_list.append(self.G(x_real, c_trg))
                for c_rafd in c_rafd_list:
                    c_trg = torch.cat([zero_celeba, c_rafd, mask_rafd], dim=1)
                    x_fake_list.append(self.G(x_real, c_trg))

                # Save the translated images.
                x_concat = torch.cat(x_fake_list, dim=3)
                result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i + 1))
                save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                print('Saved real and fake images into {}...'.format(result_path))