import sys
import os
import re
import importlib
import argparse
import torch

def apply_patch(module_path, module_name, config):
    """根据配置动态应用猴子补丁"""

    # 1. 将模块所在目录添加到 sys.path
    module_dir = os.path.dirname(module_path)
    print(f"[Patcher] Adding '{module_dir}' to system path to import '{module_name}'.")
    sys.path.insert(0, module_dir)

    # 2. 动态导入用户指定的模块和类
    print(f"[Patcher] Loading module '{module_name}'...")
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        print(f"[Patcher] Error: Module '{module_name}' not found or failed to import.")
        print(f"Import error: {e}")
        sys.exit(1)

    if not hasattr(module, config.class_name):
        print(f"[Patcher] Error: Class '{config.class_name}' not found in module '{module_name}'.")
        sys.exit(1)
    
    TrainerClass = getattr(module, config.class_name)
    print(f"[Patcher] Found class '{config.class_name}'.")

    # 3. 保存原始方法
    original_epoch_method = getattr(TrainerClass, config.epoch_method)
    original_fit_method = getattr(TrainerClass, config.fit_method)
    print(f"[Patcher] Preparing to patch methods '{config.fit_method}' and '{config.epoch_method}'.")

    # 4. 创建和应用补丁方法
    # --- 检查点相关函数 ---
    CHECKPOINT_DIR = config.ckpt_dir
    SAVE_INTERVAL = config.save_interval
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    def find_latest_checkpoint():
        files = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("checkpoint_epoch_") and f.endswith(".pth")]
        if not files:
            return None
        valid = [(int(m.group(1)), f) for f in files if (m := re.match(r"checkpoint_epoch_(\d+)\.pth", f))]
        if not valid:
            return None
        valid.sort()
        return os.path.join(CHECKPOINT_DIR, valid[-1][1])

    # --- 补丁方法定义 ---
    # TODO 如果没有训练一轮的函数呢？
    def patched_epoch_method(self, epoch, *args, **kwargs):
        # 调用原始的单轮训练方法
        original_epoch_method(self, epoch, *args, **kwargs)
        
        # 保存检查点
        if epoch % SAVE_INTERVAL == 0:
            checkpoint = {
                "model_state_dict": getattr(self, config.model_attr).state_dict(),
                "optimizer_state_dict": getattr(self, config.optimizer_attr).state_dict(),
                "scheduler_state_dict": getattr(self, config.scheduler_attr).state_dict(),
                "epoch": epoch,
                "best_metric": getattr(self, config.best_metric_attr, 0.0),
            }
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch:04d}.pth")
            torch.save(checkpoint, ckpt_path)
            print(f"[Patcher] Checkpoint saved at {ckpt_path}")

    # TODO 通用性不强，看能不能通过__init__方法注入检查点状态进行恢复
    def patched_fit_method(self, *args, **kwargs):
        # 恢复检查点
        latest_ckpt = find_latest_checkpoint()
        start_epoch = 1
        device = getattr(self, 'device', 'cpu') # 尝试获取device属性，否则默认为cpu

        if latest_ckpt:
            print(f"[Patcher] Found checkpoint: {latest_ckpt}, loading...")
            ckpt = torch.load(latest_ckpt, map_location=device)
            getattr(self, config.model_attr).load_state_dict(ckpt["model_state_dict"])
            getattr(self, config.optimizer_attr).load_state_dict(ckpt["optimizer_state_dict"])
            getattr(self, config.scheduler_attr).load_state_dict(ckpt["scheduler_state_dict"])
            setattr(self, config.best_metric_attr, ckpt.get("best_metric", 0.0))
            start_epoch = ckpt.get("epoch", 0) + 1
            print(f"[Patcher] Resume from epoch {start_epoch}")
        else:
            print("[Patcher] No checkpoint found, training from scratch.")

        # TODO 当前的方法虽然具备一定通用性，但丢失其他逻辑（只能保证基本的训练过程）
        # 注意：这里的循环逻辑是从原 `fit` 方法中简化并泛化的。
        # 它假设原始的 `fit` 方法的核心是一个从1到N的epoch循环。
        # 这种重写方式使得它不依赖于原始 `fit` 方法的内部实现，但可能丢失一些非核心逻辑。
        total_epochs = getattr(self, config.epochs_attr)
        print(f"Starting training process from epoch {start_epoch} to {total_epochs}...")
        
        # 直接调用原始的fit方法，但需要修改其循环行为，这里我们选择重写循环
        # 这种方式更灵活，但也意味着原始fit方法中循环之外的逻辑不会被执行
        # 一个更优的方案是让用户确保他们的fit方法可以接受一个start_epoch参数
        for epoch in range(start_epoch, total_epochs + 1):
            # 获取单轮训练方法并调用
            epoch_runner = getattr(self, config.epoch_method)
            epoch_runner(epoch)

            # 这里我们就不再重复实现 validation 和 learning_rate.step() 的逻辑了
            # 我们假设这些逻辑要么在 epoch_method 内部，要么在原始的 fit 方法的其他地方
            # 这个简化的 `patched_fit` 专注于恢复和循环
            # 在我们的案例中，`lr_scheduler.step()` 在原始 `fit` 循环里，所以这里需要加上
            getattr(self, config.scheduler_attr).step()

        print("Finished training.")


    # --- 应用补丁 ---
    setattr(TrainerClass, config.epoch_method, patched_epoch_method)
    setattr(TrainerClass, config.fit_method, patched_fit_method)
    print("[Patcher] Methods patched successfully.")

    # 5. 调用原始模块的主函数
    if hasattr(module, 'main'):
        print("[Patcher] Starting original 'main' function...")
        module.main()
    else:
        print(f"[Patcher] Error: 'main' function not found in module '{config.module_name}'. Cannot start training.")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A configurable monkey patcher for adding checkpointing to training scripts.")
    
    # 补丁脚本自身参数
    patch_group = parser.add_argument_group('Patcher Configuration')
    patch_group.add_argument("--ckpt-dir", type=str, default="checkpoints", help="Directory to save checkpoints.")
    patch_group.add_argument("--save-interval", type=int, default=1, help="Save checkpoint every N epochs.")

    # 用户需要配置的名称
    config_group = parser.add_argument_group('Target Code Configuration')
    config_group.add_argument("--module-path", type=str, required=True, help="Path to the training script file (e.g., 'train.py' or '../src/train.py').")
    config_group.add_argument("--class-name", type=str, required=True, help="Name of the Trainer class.")
    config_group.add_argument("--fit-method", type=str, default="fit", help="Name of the main training loop method.")
    config_group.add_argument("--epoch-method", type=str, default="_train_one_epoch", help="Name of the single epoch training method.")
    config_group.add_argument("--model-attr", type=str, default="model", help="Attribute name for the model in the trainer.")
    config_group.add_argument("--optimizer-attr", type=str, default="optimizer", help="Attribute name for the optimizer.")
    config_group.add_argument("--scheduler-attr", type=str, default="lr_scheduler", help="Attribute name for the LR scheduler.")
    config_group.add_argument("--epochs-attr", type=str, default="epochs", help="Attribute name for the total number of epochs.")
    config_group.add_argument("--best-metric-attr", type=str, default="best_map", help="Attribute name for the best metric score.")

    args, remaining = parser.parse_known_args()
    
    # 将'--'后面的参数传递给子脚本
    if '--' in remaining:
        try:
            separator_index = remaining.index('--')
            remaining = remaining[separator_index + 1:]
        except ValueError:
            # 如果'--'不在列表中，则什么也不做。这理论上不会发生。
            pass

    script_name = sys.argv[0]
    sys.argv = [script_name] + remaining
    
    # 从文件路径中解析出目录和模块名
    module_path_str = os.path.abspath(args.module_path)
    module_name_str = os.path.splitext(os.path.basename(module_path_str))[0]

    if not os.path.exists(module_path_str):
        print(f"[Patcher] Error: Module path '{module_path_str}' does not exist.")
        sys.exit(1)

    # 调试：逐行打印接收到的所有参数
    print("接收到的参数:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("剩余参数:")
    for arg in remaining:
        print(f"  {arg}")

    apply_patch(module_path_str, module_name_str, args)

# python patch/patch_configurable.py --module-path train.py --class-name Trainer --ckpt-dir configurable_checkpoints --save-interval 5 -- --lr 0.002 --momentum 0.9
