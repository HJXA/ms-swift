from typing_extensions import override
from typing import TYPE_CHECKING, Any, Optional, Union, List, Callable
import torch
import torch.nn as nn
from peft import PeftModel
from transformers.utils import is_peft_available
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
import os
import pickle


from swift.utils import get_logger,get_model_parameter_info
from swift.trainers.seq2seq_trainer import Seq2SeqTrainer_Swift
from swift.trainers.utils import per_token_loss_func, per_token_loss_func_sp

import threading
import queue
import re
import time

import sys
sys.path.append("/ruilab/jxhe/CoE_Monitor/utils")

from Layer_Hidden import Layer_Hidden_Train
from Coe_Scores import CoEScoreInfo_Train as CoEScoreInfo

logger = get_logger()

class CoETrainer(Seq2SeqTrainer_Swift):
    r"""
    继承自 CustomSeq2SeqTrainer, 用于实现自定义的 CoeLoss。
    只覆盖了 compute_loss 方法，保留了父类的所有初始化和辅助功能。
    """
    def __init__(self, test_falg = False,CoE_Flag=True, *args, **kwargs):
        super().__init__(*args, **kwargs)


        self.CoE_Flag=CoE_Flag
        self.Coe = []
        self.step_count = 0 


        

        self.input = None
        self.output = None
        self.test_falg = test_falg

        if CoE_Flag:

            if not hasattr(self, 'args') or not self.args.resume_from_checkpoint: # /ruilab/jxhe/CoE_Monitor/ms-swift/coe_train_result/PT_HJXA_Llama_104M_Minimind/v0-20260309-213557/checkpoint-34000

                self.save_layer_hidden_root = os.path.join(
                    self.args.output_dir.replace("output", "coe_train_result"),
                    "Layer_Hidden_Train"
                )
            else:
                try:
                    base = self.args.resume_from_checkpoint.replace("output", "coe_train_result")

                    self.save_layer_hidden_root = os.path.join(
                        os.path.dirname(base),
                        "Layer_Hidden_Train"
                    )
                    print(f"[CoeTrainer] 从 checkpoint 路径构建 CoE 保存路径: {self.save_layer_hidden_root}")
                except Exception as e:
                    print(f"[CoeTrainer] 从 checkpoint 路径构建 CoE 保存路径失败: {e}")
                    self.save_layer_hidden_root = os.path.join(
                        self.args.output_dir.replace("output", "coe_train_result"),
                        "Layer_Hidden_Train"
                    )
                    print(f"[CoeTrainer] 已使用默认路径: {self.save_layer_hidden_root}")
            os.makedirs(self.save_layer_hidden_root, exist_ok=True)


            self._try_load_coe_state()


            # 参数训练信息
            if self.accelerator.is_main_process:

                param_stats = get_model_parameter_info(self.model)

                print(param_stats)


            print(f"[CoeTrainer] Rank {self.accelerator.process_index} 初始化完成")

            # ========异步保存===========
    #         self._save_queue = queue.Queue(maxsize=32) 
    #         def _async_save_worker():
    #             while True:
    #                 item = self._save_queue.get()
    #                 if item is None:
    #                     break
                    
    #                 try:
    #                     task_type = item.get("type")
                        
    #                     if task_type == "tensor":
    #                         # 处理 Tensor 保存
    #                         torch.save(item["data"], item["path"])
                            
    #                     elif task_type == "coe":
    #                         # 处理 CoE 保存
    #                         with open(item["path"], "wb") as f:
    #                             pickle.dump(item["data"], f)
                                
    #                         # 顺便在后台处理文件删除，避免阻塞
    #                         for rm_path in item.get("remove_paths", []):
    #                             if os.path.exists(rm_path):
    #                                 os.remove(rm_path)
    #                 except Exception as e:
    #                     print(f"[AsyncWorker] 保存出错: {e}")
    #                 finally:
    #                     self._save_queue.task_done()

    #         self._save_thread = threading.Thread(
    #             target=_async_save_worker,
    #             daemon=True
    #         )
    #         self._save_thread.start()

    # def _submit_async_layer_save(self, tensor, path):
    #     """提交 Tensor 保存任务"""
    #     self._save_queue.put({"type": "tensor", "data": tensor, "path": path})

    # def _submit_async_coe_save(self, coe_data, path, remove_paths):
    #     """提交 CoE 列表保存及旧文件清理任务"""
    #     self._save_queue.put({
    #         "type": "coe", 
    #         "data": coe_data, 
    #         "path": path, 
    #         "remove_paths": remove_paths
    #     })




    def _try_load_coe_state(self):
        """
        尝试从保存目录中加载最新的 Coe 和 step_count
        """
        # 1. 构建保存路径 (逻辑需与保存代码完全一致)
        # 注意：这里假设 self.args.output_dir 已经被父类初始化
        if not hasattr(self, 'args') or not self.args.resume_from_checkpoint:
            print("[CoeTrainer] args 中未设置 resume_from_checkpoint，无法加载 CoE 状态。")
            return

        resume_dir = self.args.resume_from_checkpoint
        print(f"[CoeTrainer] 尝试从 {resume_dir} 加载步数...")

        rank = self.accelerator.process_index

        dirname = os.path.basename(resume_dir)   # checkpoint-34000

        pattern = re.compile(r"checkpoint-(\d+)")
        match = pattern.match(dirname)

        max_step = 0
        if match:
            max_step = int(match.group(1))

        self.step_count = max_step

        print(f"[CoeTrainer] Rank {rank}: 步数已设置为 {max_step}")
        


    @override
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):


        if self.test_falg and self.accelerator.is_main_process:
            print("============测试模式=============")
            self.input = inputs 
            torch.cuda.synchronize()
            comput_loss_start = time.time()

        self.step_count += 1

        ## MS-Swift 原 compute_loss 代码开始 ##
        labels = None
        compute_loss_func: Callable = inputs.pop('compute_loss_func', None)
        loss_scale = inputs.pop('loss_scale', None)
        text_position_ids = inputs.pop('text_position_ids', None)
        if text_position_ids is None:
            text_position_ids = inputs.get('position_ids')
        channels = inputs.pop('channel', None)

        if (self.label_smoother is not None or compute_loss_func is not None or loss_scale is not None
                or self.args.enable_dft_loss or self.args.enable_channel_loss
                or self.template.sequence_parallel_size > 1) and 'labels' in inputs:
            if self.args.use_liger_kernel:
                logger.warning_once('The cross_entropy loss function defined in Liger Kernel will not '
                                    'take effect, potentially leading to increased GPU memory consumption.')
            labels = inputs.pop('labels')
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        if getattr(outputs, 'aux_loss', None) is not None:
            mode = 'train' if self.model.training else 'eval'
            self.custom_metrics[mode]['aux_loss'].update(outputs.aux_loss)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if hasattr(self.args, 'past_index') and self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is None:
            labels = inputs['labels']
            outputs.loss = outputs.loss.to(labels.device)
            # fix https://github.com/huggingface/transformers/issues/34263
            if num_items_in_batch is not None:
                outputs.loss = outputs.loss * ((labels[:, 1:] != -100).sum() / num_items_in_batch)

            if isinstance(outputs, dict) and 'loss' not in outputs:
                raise ValueError(
                    'The model did not return a loss from the inputs, only the following keys: '
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}.")
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]
        else:
            outputs.loss = None
            if (self.args.enable_dft_loss or loss_scale is not None or self.args.enable_channel_loss
                    or self.template.sequence_parallel_size > 1):
                if self.template.sequence_parallel_size > 1:
                    outputs.loss = per_token_loss_func_sp(outputs, labels, enable_dft_loss=self.args.enable_dft_loss)
                else:
                    outputs.loss = per_token_loss_func(outputs, labels, enable_dft_loss=self.args.enable_dft_loss)

                if loss_scale is not None:
                    loss_scale = torch.roll(loss_scale, shifts=-1, dims=-1).view(-1)
                    outputs.loss = outputs.loss * loss_scale

                if self.args.enable_channel_loss and channels is not None:
                    mode = 'train' if self.model.training else 'eval'
                    metrics = self.custom_metrics[mode]
                    masks = torch.roll(labels, shifts=-1, dims=-1).view(-1) != -100
                    if self.template.padding_free:
                        cu_seqlens = self.get_cu_seqlens(text_position_ids, inputs.get('logits_to_keep'))
                    else:
                        cu_seqlens = torch.arange(0, labels.shape[0] + 1) * labels.shape[1]
                    for i in range(cu_seqlens.shape[0] - 1):
                        channel = channels[i]
                        slice_ = slice(cu_seqlens[i], cu_seqlens[i + 1])
                        metrics[f'loss_{channel}'].update(outputs.loss[slice_][masks[slice_]])

            unwrapped_model = self.accelerator.unwrap_model(model)
            if is_peft_available() and isinstance(unwrapped_model, PeftModel):
                model_name = unwrapped_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            # User-defined compute_loss function
            if compute_loss_func is not None:
                loss = compute_loss_func(
                    outputs, labels, num_items_in_batch=num_items_in_batch, loss_scale=loss_scale, trainer=self)
            elif self.label_smoother is None:
                # Handle the outputs.loss generated by loss_scale.
                if num_items_in_batch is None:
                    num_items_in_batch = (labels[:, 1:] != -100).sum()
                loss = outputs.loss.sum() / num_items_in_batch
            else:
                if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                    loss = self.label_smoother(outputs, labels, shift_labels=True)
                else:
                    loss = self.label_smoother(outputs, labels)

            if self.model.model_info.is_moe_model and self.args.router_aux_loss_coef is not None:
                aux_loss = outputs.get('aux_loss')
                if aux_loss is not None:
                    if num_items_in_batch is not None:
                        aux_loss = aux_loss * ((labels[:, 1:] != -100).sum() / num_items_in_batch)
                    loss = loss + self.args.router_aux_loss_coef * aux_loss.to(loss.device)

        if getattr(self.args, 'average_tokens_across_devices',
                   False) and self.model_accepts_loss_kwargs and num_items_in_batch is not None:
            loss *= self.accelerator.num_processes

        if (outputs.logits is not None and labels is not None and self.args.tuner_backend != 'unsloth'):
            cu_seqlens = None
            if self.template.padding_free and self.args.acc_strategy == 'seq':
                cu_seqlens = self.get_cu_seqlens(text_position_ids, inputs.get('logits_to_keep'))
            # Liger does not have logits
            # Unsloth has a bug with output logits
            self._compute_acc(outputs, labels, cu_seqlens=cu_seqlens)

        ### MS-Swift 原 compute_loss 代码结束 ###


        if self.test_falg and self.accelerator.is_main_process:
            torch.cuda.synchronize()
            print("raw_compute_loss",time.time()-comput_loss_start)
            coe_start = time.time()
            print("labels 的形状",labels.shape)

            print(f"labels 中 -100 比例",(labels == -100).float().mean())

        if not self.CoE_Flag:
            return (loss, outputs) if return_outputs else loss


        current_labels = labels if labels is not None else inputs.get('labels')

        if current_labels is None:
            print("================\nWarning: current_labels is None.\n================")




        layer_hidden_state = Layer_Hidden_Train(outputs.hidden_states, labels = current_labels, steps = self.step_count)
        current_labels = None  # 释放内存

        if self.test_falg and self.accelerator.is_main_process:
            self.output = outputs 

        outputs.hidden_states = None  # 释放内存

        rank = self.accelerator.process_index
        step = self.step_count

        save_path = os.path.join(
            self.save_layer_hidden_root,
            f"Step{step}_Rank{rank}.pt"
        )

        tensor_to_save = (
            layer_hidden_state
            .detach()
            .to(torch.bfloat16)   # 强烈建议压缩
            .cpu()
        )

        # self._submit_async_layer_save(tensor_to_save, save_path)

        if self.test_falg and self.accelerator.is_main_process:
            torch.cuda.synchronize()
            save_layer_start = time.time()
        torch.save(tensor_to_save, save_path)
        if self.test_falg and self.accelerator.is_main_process:
            torch.cuda.synchronize()
            print("save_layer_time",time.time()-save_layer_start)
        tensor_to_save = None


        batch_size = layer_hidden_state.shape[0]

        # total_mag = 0.0
        total_ang = 0.0
        total_last_ang = 0.0
        total_z_a = 0.0
        total_ang_sum = 0.0


        for i in range(batch_size):
            output_L_coe = CoEScoreInfo(layer_hidden_state[i]) # (Layer,D)
            

            # coe_output_M = output_L_coe.compute_CoE_Mag()
            coe_output_A = output_L_coe.compute_CoE_Ang()


            # val_m = coe_output_M[1].detach().cpu().item()
            val_a = coe_output_A[1].detach().cpu().item()
            coe_output_z_A = coe_output_A[3].detach().cpu().item()
            ang_sum = coe_output_A[4].detach().cpu().item()


            # total_mag += val_m
            total_ang += val_a
            total_z_a += coe_output_z_A
            total_ang_sum += ang_sum


            coe_last_ang = output_L_coe.compute_Last_Ang()
            val_last_ang = coe_last_ang.detach().cpu().item()
            total_last_ang += val_last_ang

        output_L_coe = None  # 释放内存

        # avg_mag = total_mag / batch_size
        avg_ang = total_ang / batch_size
        avg_last_ang = total_last_ang / batch_size
        avg_z_a = total_z_a / batch_size
        avg_ang_sum = total_ang_sum / batch_size

           

        metrics = {
            # "CoE/Avg_Mag": avg_mag,
            "CoE/Ang": avg_ang,
            "CoE/Z_A": avg_z_a,
            "CoE/Last_Ang": avg_last_ang,
            "CoE/Ang_Sum": avg_ang_sum
        }
        
        if rank == 0:
            args = self.args
            if 'swanlab' in args.report_to:
                import swanlab
                swanlab.log(metrics, step=self.step_count)

        layer_hidden_state = None  # 释放内存

        if self.test_falg and self.accelerator.is_main_process:
            torch.cuda.synchronize()
            print("coe_add",time.time()-coe_start)
            

        return (loss, outputs) if return_outputs else loss


    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`dict[str, torch.Tensor | Any]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        # Prepare buffers for context parallelism

        if self.test_falg and self.accelerator.is_main_process:
            torch.cuda.synchronize()
            step_start = time.time()

        loss = super().training_step(model, inputs, num_items_in_batch)

        if self.test_falg and self.accelerator.is_main_process:
            torch.cuda.synchronize()
            print("step_time=",time.time()-step_start)

        return loss


        








