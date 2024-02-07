import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import utils
from models import Clipper, BrainNetwork

# tf32 data type is faster than standard float32
torch.backends.cuda.matmul.allow_tf32 = True

# multi-GPU config
from accelerate import Accelerator
accelerator = Accelerator(split_batches=False,mixed_precision='fp16')  
print("PID of this process =",os.getpid())
print = accelerator.print # only print if local_rank=0
device = accelerator.device
print("device:",device)
num_devices = torch.cuda.device_count()
if num_devices==0: num_devices = 1
num_workers = num_devices
print(accelerator.state)
local_rank = accelerator.state.local_process_index
world_size = accelerator.state.num_processes
distributed = not accelerator.state.distributed_type == 'NO'
print("distributed =",distributed, "num_devices =", num_devices, "local rank =", local_rank, "world size =", world_size)

# configurations
parser = argparse.ArgumentParser(description="Model Training Configuration")
parser.add_argument("--model_name", type=str, default="dream_rvac_training_demo", help="name of model, used for ckpt saving")
parser.add_argument("--data_path", type=str, default="nsd_data", help="Path to where NSD data is stored / where to download it to")
parser.add_argument("--subj",type=int, default=1, choices=[1, 2, 5, 7])
parser.add_argument("--batch_size", type=int, default=320, help="Batch size for training")
parser.add_argument("--hidden", action=argparse.BooleanOptionalAction, default=False, help="if True, CLIP embeddings are from last hidden layer (257x768) rather than final layer (1x768)")
parser.add_argument("--clip_variant", type=str, default="ViT-L/14", choices=["RN50", "ViT-L/14", "ViT-B/32", "RN50x64"])
parser.add_argument("--norm_embs", action=argparse.BooleanOptionalAction, default=False, help="Do l2-norming of CLIP embeddings")
parser.add_argument("--use_image_aug", action=argparse.BooleanOptionalAction, default=True, help="whether to use image augmentation",)
parser.add_argument("--num_epochs", type=int, default=240, help="number of epochs of training")
parser.add_argument("--plot_umap", action=argparse.BooleanOptionalAction, default=False, help="Plot UMAP plots alongside reconstructions")
parser.add_argument("--lr_scheduler_type", type=str, default='cycle', choices=['cycle', 'linear'])
parser.add_argument("--ckpt_saving", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--ckpt_interval", type=int, default=5, help="save backup ckpt and reconstruct every x epochs",)
parser.add_argument("--save_at_end", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_lr", type=float, default=3e-4)
parser.add_argument("--use_projector", action=argparse.BooleanOptionalAction, default=True)
args = parser.parse_args()

# create global variables without the args prefix
for attribute_name in vars(args).keys():
    globals()[attribute_name] = getattr(args, attribute_name)
    
# need non-deterministic CuDNN for conv3D to work
utils.seed_everything(seed, cudnn_deterministic=False)

# change learning rate based on number of devices
max_lr *= accelerator.num_processes
    
# change batch size based on number of devices if using multi-gpu
batch_size *= accelerator.num_processes

# change num_epochs based on number of devices if using multi-gpu
num_epochs *= accelerator.num_processes

outdir = os.path.abspath(f'../train_logs/{model_name}')
if not os.path.exists(outdir):
    os.makedirs(outdir,exist_ok=True)
if use_image_aug:
    import kornia
    from kornia.augmentation.container import AugmentationSequential
    img_augment = AugmentationSequential(
        kornia.augmentation.RandomResizedCrop((224,224), (0.6,1), p=0.3),
        kornia.augmentation.Resize((224, 224)),
        kornia.augmentation.RandomHorizontalFlip(p=0.5),
        kornia.augmentation.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.3),
        kornia.augmentation.RandomGrayscale(p=0.3),
        data_keys=["input"],
    )

# prepare models and data loaders
print('Pulling NSD webdataset data...')

train_url = "{" + f"{data_path}/webdataset_avg_split/train/train_subj0{subj}_" + "{0..17}.tar," + f"{data_path}/webdataset_avg_split/val/val_subj0{subj}_0.tar" + "}"
val_url = f"{data_path}/webdataset_avg_split/test/test_subj0{subj}_" + "{0..1}.tar"
print(train_url,"\n",val_url)
meta_url = f"{data_path}/webdataset_avg_split/metadata_subj0{subj}.json"
num_train = 8559 + 300
num_val = 982

print('Prepping train and validation dataloaders...')
train_dl, val_dl, num_train, num_val = utils.get_dataloaders(
    batch_size,'images',
    num_devices=num_devices,
    num_workers=num_workers,
    train_url=train_url,
    val_url=val_url,
    meta_url=meta_url,
    num_train=num_train,
    num_val=num_val,
    val_batch_size=300,
    cache_dir=data_path, #"/tmp/wds-cache",
    seed=seed,
    voxels_key='nsdgeneral.npy',
    to_tuple=["voxels", "images", "coco"],
    local_rank=local_rank,
    world_size=world_size,
)

print('Creating Clipper...')
clip_sizes = {"RN50": 1024, "ViT-L/14": 768, "ViT-B/32": 512, "ViT-H-14": 1024}
clip_size = clip_sizes[clip_variant]
if hidden:
    print("Using hidden layer CLIP space")
    clip_extractor = Clipper(clip_variant, device=device, hidden_state=True, norm_embs=norm_embs)
    out_dim = 77 * clip_size
else:
    print("Using final layer CLIP space")
    clip_extractor = Clipper(clip_variant, device=device, hidden_state=False, norm_embs=norm_embs)
    out_dim = clip_size
print("out_dim:", out_dim)

print('Creating voxel2clip...')
voxels_per_subj = {1: 15724, 2: 14278, 3: 15226, 4: 13153, 5: 13039, 6: 17907, 7: 12682, 8: 14386}
num_voxels = voxels_per_subj.get(subj)
voxel2clip_kwargs = dict(in_dim=num_voxels, out_dim=out_dim, clip_size=clip_size, use_projector=use_projector)
voxel2clip = BrainNetwork(**voxel2clip_kwargs)
    
print("params of voxel2clip:")
if local_rank==0:
    utils.count_params(voxel2clip)
voxel2clip.requires_grad_(True)

no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
opt_grouped_parameters = [
    {'params': [p for n, p in voxel2clip.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
    {'params': [p for n, p in voxel2clip.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = torch.optim.AdamW(opt_grouped_parameters, lr=max_lr)

global_batch_size = batch_size * num_devices
if lr_scheduler_type == 'linear':
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        total_iters=int(num_epochs*(num_train//global_batch_size)),
        last_epoch=-1
    )
elif lr_scheduler_type == 'cycle':
    total_steps=int(num_epochs*(num_train//global_batch_size))
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=max_lr,
        total_steps=total_steps,
        final_div_factor=1000,
        last_epoch=-1, pct_start=2/num_epochs
    )
    
def save_ckpt(tag):    
    ckpt_path = outdir+f'/{tag}.pth'
    print(f'saving {ckpt_path}',flush=True)
    unwrapped_model = accelerator.unwrap_model(voxel2clip)
    try:
        torch.save({
            'epoch': epoch,
            'model_state_dict': unwrapped_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'train_losses': losses,
            'val_losses': val_losses,
            'lrs': lrs,
            }, ckpt_path)
    except:
        print("Couldn't save... moving on to prevent crashing.")
    del unwrapped_model
        
print("\nDone with model preparations!")

# load clip target text
if hidden:
    clip_txt_path = 'nsd_data/clip_caps_hidden.pt'
else:
    clip_txt_path = 'nsd_data/clip_caps_final.pt'
if not os.path.isfile(clip_txt_path):
    print('\nGenerating/Loading CLIP text embedding...')
    annots_cur = np.load('nsd_data/COCO_73k_annots_curated.npy')
    clip_caps = torch.zeros((len(annots_cur), out_dim)).float().to(device)
    with torch.no_grad():
        for i, annots in enumerate(annots_cur):
            caps = list(annots[annots!=''])
            clip_text = clip_extractor.embed_text(caps).float()
            clip_caps[i] = clip_text.mean(0)
    torch.save(clip_caps, clip_txt_path)
else:
    print('\nLoading CLIP text embedding...')
    clip_caps = torch.load(clip_txt_path)
print("clip_caps.shape:", clip_caps.shape)

# main loop for training
epoch = 0
losses, val_losses, lrs = [], [], []
nce_losses, val_nce_losses = [], []
sim_losses, val_sim_losses = [], []
best_val_loss = 1e9

voxel2clip, optimizer, train_dl, val_dl, lr_scheduler = accelerator.prepare(
voxel2clip, optimizer, train_dl, val_dl, lr_scheduler
)

print(f"{model_name} starting with epoch {epoch} / {num_epochs}")
progress_bar = tqdm(range(epoch, num_epochs), ncols=1200, disable=(local_rank!=0))

for epoch in progress_bar:
    voxel2clip.train()

    sims_base = 0.
    val_sims_base = 0.
    fwd_percent_correct = 0.
    bwd_percent_correct = 0.
    val_fwd_percent_correct = 0.
    val_bwd_percent_correct = 0.
    loss_nce_sum = 0.
    val_loss_nce_sum = 0.

    for train_i, (voxel, image, coco) in enumerate(train_dl):
        with torch.cuda.amp.autocast():
            optimizer.zero_grad()

            repeat_index = train_i % 3
            voxel = voxel[:,repeat_index].float()

            image = img_augment(image) if use_image_aug else image

            voxel, perm, betas, select = utils.mixco(voxel)

            clip_target_img = clip_extractor.embed_image(image).float()  
            clip_target_txt = clip_caps[coco.squeeze()].float()

            _, clip_voxels_proj = voxel2clip(voxel)
            
            clip_voxels_norm = F.normalize(clip_voxels_proj.flatten(1), dim=-1)
            clip_target_img_norm = F.normalize(clip_target_img.flatten(1), dim=-1)
            clip_target_txt_norm = F.normalize(clip_target_txt.flatten(1), dim=-1)

            if hidden:
                loss_nce = utils.mixco_nce(
                    clip_voxels_norm,
                    clip_target_txt_norm,
                    temp=.006, 
                    perm=None, betas=None, select=None) 
            else:
                loss_nce_i = utils.mixco_nce(
                    clip_voxels_norm,
                    clip_target_img_norm,
                    temp=.006, 
                    perm=perm, betas=betas, select=select)

                loss_nce_t = utils.mixco_nce(
                    clip_voxels_norm,
                    clip_target_txt_norm,
                    temp=.006, 
                    perm=perm, betas=betas, select=select)
                loss_nce = loss_nce_t + loss_nce_i
                  
            loss_nce_sum += loss_nce.item()
            loss = loss_nce #+ mse_loss(clip_target_txt_norm, clip_voxels_norm)
            utils.check_loss(loss)
            
            accelerator.backward(loss)
            optimizer.step()

            losses.append(loss.item())
            lrs.append(optimizer.param_groups[0]['lr'])

            sims_base += F.cosine_similarity(clip_target_txt_norm, clip_voxels_norm).mean().item()

            # forward and backward top 1 accuracy        
            labels = torch.arange(len(clip_target_txt_norm)).to(device) 
            fwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_voxels_norm, clip_target_txt_norm), labels, k=1)
            bwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_target_txt_norm, clip_voxels_norm), labels, k=1)

            if lr_scheduler_type is not None:
                lr_scheduler.step()

    voxel2clip.eval()
    for val_i, (voxel, image, coco) in enumerate(val_dl): 
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                # repeat_index = val_i % 3
                # voxel = voxel[:,repeat_index].float()
                voxel = torch.mean(voxel, axis=1).float()

                clip_target_img = clip_extractor.embed_image(image).float()
                clip_target_txt = clip_caps[coco.squeeze()].float()

                clip_voxels, clip_voxels_proj = voxel2clip(voxel)
                if hidden:
                    clip_voxels = clip_voxels.view(len(voxel), -1, clip_size)
                
                aligned_clip_voxels = clip_voxels

                clip_voxels_norm = F.normalize(clip_voxels_proj.flatten(1), dim=-1)
                clip_target_img_norm = F.normalize(clip_target_img.flatten(1), dim=-1)
                clip_target_txt_norm = F.normalize(clip_target_txt.flatten(1), dim=-1)

                val_loss_nce_i = utils.mixco_nce(
                    clip_voxels_norm,
                    clip_target_img_norm,
                    temp=.006, 
                    perm=None, betas=None, select=None)

                val_loss_nce_t = utils.mixco_nce(
                    clip_voxels_norm,
                    clip_target_txt_norm,
                    temp=.006, 
                    perm=None, betas=None, select=None)
                
                val_loss_nce = val_loss_nce_t + val_loss_nce_i
                val_loss_nce_sum += val_loss_nce.item()
                val_loss = val_loss_nce
                utils.check_loss(val_loss)
                val_losses.append(val_loss.item())
                
                val_sims_base += F.cosine_similarity(clip_target_txt_norm, clip_voxels_norm).mean().item()
                
                labels = torch.arange(len(clip_target_txt_norm)).to(device)
                val_fwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_voxels_norm, clip_target_txt_norm), labels, k=1)
                val_bwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_target_txt_norm, clip_voxels_norm), labels, k=1)

    if local_rank==0:        
        if (not save_at_end and ckpt_saving) or (save_at_end and epoch == num_epochs - 1):
            # save best model
            val_loss = np.mean(val_losses[-(val_i+1):])
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_ckpt('best')
            else:
                print(f'not best - val_loss: {val_loss:.3f}, best_val_loss: {best_val_loss:.3f}')
                
        if utils.is_interactive():
            clear_output(wait=True)
            
        logs = {"train/loss": np.mean(losses[-(train_i+1):]),
            "val/loss": np.mean(val_losses[-(val_i+1):]),
            "train/lr": lrs[-1],
            "train/num_steps": len(losses),
            "val/num_steps": len(val_losses),
            "train/cosine_sim_base": sims_base / (train_i + 1),
            "val/cosine_sim_base": val_sims_base / (val_i + 1),
            "train/fwd_pct_correct": fwd_percent_correct / (train_i + 1),
            "train/bwd_pct_correct": bwd_percent_correct / (train_i + 1),
            "val/val_fwd_pct_correct": val_fwd_percent_correct / (val_i + 1),
            "val/val_bwd_pct_correct": val_bwd_percent_correct / (val_i + 1),
            "train/loss_nce": loss_nce_sum / (train_i + 1),
            "val/loss_nce": val_loss_nce_sum / (val_i + 1)}
        progress_bar.set_postfix(**logs)

        # Save model checkpoint and reconstruct
        save_ckpt(f'last')
        if epoch % ckpt_interval == 0:
            save_ckpt(f'last_backup')
            if plot_umap:
                import umap
                print('umap plotting...')
                combined = np.concatenate((clip_target_txt.flatten(1).detach().cpu().numpy(),
                                            clip_voxels_proj.flatten(1).detach().cpu().numpy()),axis=0)
                reducer = umap.UMAP(random_state=42)
                embedding = reducer.fit_transform(combined)

                colors=np.array([[0,0,1,.5] for i in range(len(clip_target_txt))])
                colors=np.concatenate((colors, np.array([[0,1,0,.5] for i in range(len(clip_voxels_proj))])))

                fig = plt.figure(figsize=(5,5))
                plt.scatter(
                    embedding[:, 0],
                    embedding[:, 1],
                    c=colors)
                plt.savefig(os.path.join(outdir, f'umap-val-epoch{epoch:03d}.png'))
                plt.show()
                        
    # wait for other GPUs to catch up if needed
    accelerator.wait_for_everyone()

print("\n===Finished!===\n")
if not utils.is_interactive():
    sys.exit(0)