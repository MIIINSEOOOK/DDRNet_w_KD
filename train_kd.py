
import os
import argparse
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from DDRNet import DDRNet
from functions import SegmentationDataset, display_dataset_info, CrossEntropy, load_state_dict, WarmupPolyEpochLR
from student_hooks import attach_layer_hook
from kd_losses import LogitsKDLoss, FeatureProjector, CosineFeatureKDLoss
from teacher import DINOv2Teacher, DINOV2_EMBED_DIM

def boolean_string(s):  # robust bool arg
    if s.lower() in {"true", "1", "yes", "y"}: return True
    if s.lower() in {"false", "0", "no", "n"}: return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

def build_teacher(args, device):
    if args.teacher_type == "dinov2":
        teacher = DINOv2Teacher(args.teacher_arch, device=device)
        teacher_dim = DINOV2_EMBED_DIM.get(args.teacher_arch, args.teacher_dim)
        return teacher, teacher_dim
    elif args.teacher_type == "seglogit":
        # Use any segmentation teacher compatible with your environment (user provides a torch.load-able module)
        if args.teacher_ckpt is None:
            raise ValueError("seglogit teacher requires --teacher_ckpt (path to checkpoint of a segmentation model)")
        # NOTE: We deliberately don't prescribe the architecture; load user-provided model file that
        # must define a `build_model(num_classes)` function which returns a nn.Module producing logits.
        import importlib.util
        spec = importlib.util.spec_from_file_location("teacher_model", args.teacher_model_py)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        t_model = mod.build_model(num_classes=args.num_classes)
        state = torch.load(args.teacher_ckpt, map_location="cpu")
        t_model.load_state_dict(state, strict=False)
        t_model.to(device).eval()
        for p in t_model.parameters():
            p.requires_grad_(False)
        return t_model, None
    else:
        return None, None

def main_worker(args):
    # ----- DDP init -----
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # ----- Data -----
    train_dataset = SegmentationDataset(args.dataset_dir, args.crop_size, 'train', args.scale_range)
    display_dataset_info(args.dataset_dir, train_dataset)
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, drop_last=True, shuffle=True)
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0, pin_memory=True)

    # ----- Student -----
    student = DDRNet(num_classes=args.num_classes).to(device)
    if args.loadpath is not None:
        # pretrain weights (e.g., ImageNet) or checkpoint resume
        state_dict = torch.load(args.loadpath, map_location={"cuda:0": f"cuda:{local_rank}"})
        load_state_dict(student, state_dict)
    student = DDP(student, device_ids=[local_rank])

    # hook from DDRNet high-res deep feature for feature KD (layer5_ output is 128ch by default)
    feat_hook = attach_layer_hook(student, "layer5_")

    # ----- Teacher -----
    teacher, teacher_dim = build_teacher(args, device)

    # projector for feature KD (student 128ch -> teacher_dim)
    projector = None
    feature_kd = None
    if args.teacher_type == "dinov2":
        # DDRNet layer5_ output channels: 128 (planes*2 * expansion), unless planes changed.
        # We infer it from a dummy forward pass to be safe
        dummy_in = torch.randn(1, 3, args.crop_size[0], args.crop_size[1], device=device)
        student.train()
        with torch.no_grad():
            _ = student(dummy_in)
        student_ch = feat_hook.feat.shape[1]
        if teacher_dim is None:
            # default to dinov2 small embedding
            teacher_dim = 384
        projector = FeatureProjector(student_ch, teacher_dim).to(device)
        feature_kd = CosineFeatureKDLoss(projector, ignore_index=args.ignore_label)

    # ----- Losses/Opt/Sched -----
    ce_loss = CrossEntropy(ignore_label=args.ignore_label)  # handles aux head
    kd_logits = LogitsKDLoss(temperature=args.kd_temperature, ignore_index=args.ignore_label) if args.enable_kd_logits else None

    params = list(student.parameters())
    if projector is not None:
        params += list(projector.parameters())
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = WarmupPolyEpochLR(optimizer, total_epochs=args.epochs, warmup_epochs=5, warmup_ratio=5e-4)

    # ----- Logging -----
    if local_rank == 0:
        os.makedirs(args.result_dir, exist_ok=True)
        log_path = os.path.join(args.result_dir, "log_kd.txt")
        with open(log_path, "w") as f:
            f.write("Epoch\tCE\tKD_feat\tKD_logit\tTotal\tLR\n")

    # ----- Train -----
    min_total = 1e9
    for epoch in range(args.epochs):
        student.train()
        sampler.set_epoch(epoch)

        total_ce = total_kdf = total_kdl = total_all = 0.0
        loop = tqdm(dataloader, desc=f"[GPU {local_rank}] KD Epoch [{epoch+1}/{args.epochs}]", ncols=120) if local_rank == 0 else dataloader

        for imgs, labels in loop:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # student forward
            outputs = student(imgs)  # (main, aux) or main tensor
            if isinstance(outputs, tuple):
                student_main = outputs[0]
            else:
                student_main = outputs

            # teacher forward (no grad)
            kd_feat_loss = torch.tensor(0.0, device=device)
            kd_logit_loss = torch.tensor(0.0, device=device)

            if args.teacher_type == "dinov2":
                with torch.no_grad():
                    t_feat = teacher(imgs)  # (B, Ct, Ht, Wt)
                s_feat = feat_hook.feat  # (B, Cs, Hs, Ws)
                kd_feat_loss = feature_kd(s_feat, t_feat, labels) * args.lambda_kd_feat

            if args.teacher_type == "seglogit" and kd_logits is not None:
                with torch.no_grad():
                    t_logits = teacher(imgs)  # assumes returns (B,C,H,W)
                kd_logit_loss = kd_logits(student_main, t_logits, labels) * args.lambda_kd_logit

            ce = ce_loss(outputs, labels) * args.lambda_ce

            total_loss = ce + kd_feat_loss + kd_logit_loss
            total_loss.backward()
            optimizer.step()

            total_ce += ce.item()
            total_kdf += kd_feat_loss.item() if torch.is_tensor(kd_feat_loss) else 0.0
            total_kdl += kd_logit_loss.item() if torch.is_tensor(kd_logit_loss) else 0.0
            total_all += total_loss.item()

            if local_rank == 0:
                loop.set_postfix(ce=ce.item(),
                                 kd_feat=kd_feat_loss.item() if torch.is_tensor(kd_feat_loss) else 0.0,
                                 kd_logit=kd_logit_loss.item() if torch.is_tensor(kd_logit_loss) else 0.0,
                                 total=total_loss.item(),
                                 lr=scheduler.get_last_lr()[0])

        dist.barrier()
        scheduler.step()

        # logging + checkpointing
        if local_rank == 0:
            n = len(dataloader)
            ce_avg = total_ce / n
            kdf_avg = total_kdf / n
            kdl_avg = total_kdl / n
            all_avg = total_all / n
            with open(log_path, "a") as f:
                f.write(f"\n{epoch+1}\t{ce_avg:.4f}\t{kdf_avg:.4f}\t{kdl_avg:.4f}\t{all_avg:.4f}\t{scheduler.get_last_lr()[0]:.8f}")

            # best checkpoint on total loss
            if all_avg < min_total:
                min_total = all_avg
                ckp = {
                    "student": student.state_dict(),
                    "projector": projector.state_dict() if projector is not None else None,
                    "epoch": epoch + 1,
                    "args": vars(args),
                }
                torch.save(ckp, os.path.join(args.result_dir, "model_kd_best.pth"))

            # periodic checkpoints
            if (epoch + 1) % args.save_every == 0:
                ckp = {
                    "student": student.state_dict(),
                    "projector": projector.state_dict() if projector is not None else None,
                    "epoch": epoch + 1,
                    "args": vars(args),
                }
                torch.save(ckp, os.path.join(args.result_dir, f"model_kd_epoch{epoch+1}.pth"))

    dist.destroy_process_group()

if __name__ == "__main__":
    os.environ.setdefault("NCCL_DEBUG", "INFO")
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")

    parser = argparse.ArgumentParser()
    # data / model
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--num_classes", type=int, default=19)
    parser.add_argument("--crop_size", type=int, nargs=2, default=[512, 1024])
    parser.add_argument("--scale_range", type=float, nargs=2, default=[0.75, 1.5])
    parser.add_argument("--ignore_label", type=int, default=255)
    parser.add_argument("--loadpath", type=str, default=None, help="student preload (e.g., ImageNet) or resume")

    # teacher
    parser.add_argument("--teacher_type", type=str, default="dinov2", choices=["none", "dinov2", "seglogit"])
    parser.add_argument("--teacher_arch", type=str, default="dinov2_vits14")
    parser.add_argument("--teacher_dim", type=int, default=None, help="override DINOv2 embedding dim if unknown")
    parser.add_argument("--teacher_ckpt", type=str, default=None, help="for seglogit")
    parser.add_argument("--teacher_model_py", type=str, default=None, help="python file that defines build_model() for seglogit")

    # losses
    parser.add_argument("--lambda_ce", type=float, default=1.0)
    parser.add_argument("--lambda_kd_feat", type=float, default=1.0)
    parser.add_argument("--lambda_kd_logit", type=float, default=1.0)
    parser.add_argument("--kd_temperature", type=float, default=4.0)
    parser.add_argument("--enable_kd_logits", type=bool, default=False)

    # train
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--save_every", type=int, default=20)

    args = parser.parse_args()

    Path(args.result_dir).mkdir(parents=True, exist_ok=True)

    torch.multiprocessing.set_start_method('spawn', force=True)
    main_worker(args)
