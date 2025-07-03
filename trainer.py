import torch
import os
import time
from torchmetrics.detection import MeanAveragePrecision

import config

class Trainer:
    def __init__(self, model, optimizer, lr_scheduler, train_loader, val_loader):
        self.model = model.to(config.DEVICE)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = config.DEVICE
        self.epochs = config.EPOCHS
        self.checkpoint_dir = config.CHECKPOINT_DIR
        self.models_dir = config.MODELS_DIR
        self.best_map = 0.0
        
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
            
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

    def _train_one_epoch(self, epoch):
        self.model.train()
        total_loss, total_cls_loss, total_reg_loss = 0, 0, 0
        start_time = time.time()

        for i, (images, targets) in enumerate(self.train_loader):
            images = list(img.to(self.device) for img in images)
            
            targets_processed = []
            for t in targets:
                boxes = []
                labels = []
                # 兼容 Subset 和完整数据集的 XML 结构
                target_objects = t['annotation']['object']
                if not isinstance(target_objects, list):
                    target_objects = [target_objects]

                for obj in target_objects:
                    bndbox = obj['bndbox']
                    boxes.append([int(bndbox['xmin']), int(bndbox['ymin']), int(bndbox['xmax']), int(bndbox['ymax'])])
                    labels.append(config.VOC_CLASSES.index(obj['name']))
                
                targets_processed.append({
                    'boxes': torch.tensor(boxes, dtype=torch.float32).to(self.device),
                    'labels': torch.tensor(labels, dtype=torch.int64).to(self.device)
                })

            loss_dict = self.model(images, targets_processed)
            losses = sum(loss for loss in loss_dict.values())
            
            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()
            
            total_loss += losses.item()
            total_cls_loss += loss_dict['classification'].item()
            total_reg_loss += loss_dict['bbox_regression'].item()
            
            if (i + 1) % 20 == 0:
                print(f"Epoch [{epoch}/{self.epochs}], Step [{i+1}/{len(self.train_loader)}], Loss: {losses.item():.4f}")
        
        epoch_duration = time.time() - start_time
        current_lr = self.optimizer.param_groups[0]['lr']
        print(f"--- Epoch {epoch} Summary ---")
        print(f"Duration: {epoch_duration:.2f}s, Avg Loss: {total_loss / len(self.train_loader):.4f}, LR: {current_lr:.6f}")
    
    def _validate(self, epoch):
        self.model.eval()
        metric = MeanAveragePrecision(box_format='xyxy', max_detection_thresholds=[300])
        print(f"--- Validating on Epoch {epoch} ---")

        with torch.no_grad():
            for images, targets in self.val_loader:
                images = list(img.to(self.device) for img in images)
                predictions = self.model(images)
                
                targets_processed = []
                for t in targets:
                    boxes = []
                    labels = []
                    target_objects = t['annotation']['object']
                    if not isinstance(target_objects, list):
                        target_objects = [target_objects]

                    for obj in target_objects:
                        bndbox = obj['bndbox']
                        boxes.append([int(bndbox['xmin']), int(bndbox['ymin']), int(bndbox['xmax']), int(bndbox['ymax'])])
                        labels.append(config.VOC_CLASSES.index(obj['name']))
                    
                    targets_processed.append({
                        'boxes': torch.tensor(boxes, dtype=torch.float32).to(self.device),
                        'labels': torch.tensor(labels, dtype=torch.int64).to(self.device)
                    })
                metric.update(predictions, targets_processed)
        
        map_value = 0.0
        try:
            results = metric.compute()
            map_value = results['map'].item()
            print(f"Validation mAP: {map_value:.4f}")
        except Exception as e:
            print(f"Could not compute mAP. Error: {e}")
        
        return map_value

    def fit(self):
        print("Starting training process...")
        for epoch in range(1, self.epochs + 1):
            self._train_one_epoch(epoch)
            
            current_map = 0.0
            if self.val_loader:
                current_map = self._validate(epoch)

            if current_map > self.best_map:
                self.best_map = current_map
                best_model_path = os.path.join(self.models_dir, 'best_model.pth')
                torch.save(self.model.state_dict(), best_model_path)
                print(f"New best model saved with mAP: {self.best_map:.4f} to {best_model_path}")

            self.lr_scheduler.step()

            if epoch % 10 == 0 or epoch == self.epochs:
                checkpoint_path = os.path.join(self.checkpoint_dir, f'ssd_epoch_{epoch}.pth')
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")
        
        print("Finished training.") 