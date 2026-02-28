from typing_extensions import override
from typing import TYPE_CHECKING, Any, Optional, Union, List, Callable
import torch
import torch.nn as nn
from peft import PeftModel
from transformers.utils import is_peft_available
from transformers.trainer import _is_peft_model, OptimizerNames
from deepspeed.runtime.zero.utils import is_zero_param
from deepspeed.utils import safe_set_full_grad, safe_set_local_grad, safe_get_full_grad
from transformers.utils import (
    is_sagemaker_mp_enabled,
    is_torch_xpu_available,
    is_torch_mlu_available,
    is_torch_musa_available,
    is_torch_npu_available,
    is_torch_mps_available,
    is_torch_hpu_available,
)
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from accelerate.utils import DistributedType # 确保引入
import os
import pickle


from swift.utils import get_logger,get_model_parameter_info
from swift.trainers.trainers import Seq2SeqTrainer_SWIFT 
from swift.trainers.utils import per_token_loss_func, per_token_loss_func_sp

import sys 
# (保留你的 sys.path 引用)
sys.path.append("/data/jxhe/LLM/github/Chain-of-Embedding/My/MLLM/utils")
from Layer_Hidden import Layer_Hidden_Train
from Coe_Scores import CoEScoreInfo

logger = get_logger()

class CoETrainer(Seq2SeqTrainer_SWIFT):
    r"""
    继承自 CustomSeq2SeqTrainer, 用于实现自定义的 CoeLoss。
    只覆盖了 compute_loss 方法，保留了父类的所有初始化和辅助功能。
    """
    def __init__(self, image_token_id = None, test_falg = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
            
        self.Coe = []
        self.step_count = 0


        # 定义一个状态 Flag: True=冻结视觉(训练语言), False=冻结语言(训练视觉)
        # 默认 None，等待第一次 compute_loss 赋值
        self.freeze_vision_flag = None 
        self.count_label = False

        self.input = None
        self.output = None
        self.test_falg = test_falg
        self.image_token_id = image_token_id

            # print("Tokenizer __dict__ keys:")
            # for k in self.tokenizer.__dict__.keys():
            #     print(k)

            # print("=== special_tokens_map ===")
            # print(self.tokenizer._special_tokens_map)

            # print("=== extra_special_tokens ===")
            # print(self.tokenizer.extra_special_tokens)

        # 必须用 tokenizer 里已经注册的 token
        self.image_token_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")

        # 强校验（非常重要）
        assert self.image_token_id != self.tokenizer.unk_token_id, \
            "<|image_pad|> not found in tokenizer vocab"

        if not self.count_label and self.accelerator.is_main_process:
            print("=============MS-SwiFT MY Trainer================")
            print("image_token_id:", self.image_token_id)
            self.count_label = True

            

    def _count_model_parameters(self,model) -> int:
        """计算模型的总参数量（含冻结和非冻结参数）。"""
        if self.accelerator.is_main_process:

            param_stats = get_model_parameter_info(model)

            print(param_stats)
            
            self.count_label = True

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

        if self.test_falg:
            print("============测试模式=============")
            
            self.input = inputs 

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

        current_labels = labels if labels is not None else inputs.get('labels')

        if current_labels is None:
            print("================\nWarning: current_labels is None, cannot compute CoE scores in MS-SWIFT Mode.\n================")

        

        # print("inputs __dict__ keys:")
        # for k in inputs.keys():
        #     print(k)

        inputs_ids = inputs.get('input_ids', None)

        # if self.step_count == 1:
        #     print(len(inputs_ids[0]))
        #     print(inputs_ids[0][50:])

        save_dir = f"{self.args.output_dir.replace('output','coe_train_result')}/Layer_Hidden_Train/Step{self.step_count}_Rank{self.accelerator.process_index}"



        layer_hidden_state = Layer_Hidden_Train(outputs.hidden_states, labels = current_labels,image_token_id = self.image_token_id, inputs_ids = inputs_ids, save_dir = save_dir,steps = self.step_count)  # 调用之前定义的函数
        inputs_ids = None  # 释放内存
        current_labels = None  # 释放内存

        if self.test_falg:
            print("============测试模式=============")
            self.output = outputs 

        outputs.hidden_states = None  # 释放内存

        batch_size = layer_hidden_state.shape[0]

        # 1. 计算当前 Rank (GPU) 的局部统计
        local_seeing_count = 0 
        local_reasoning_count = 0
        select = 0 

        for i in range(batch_size):
            output_L_coe = CoEScoreInfo(layer_hidden_state[i])
            coe_output_M = output_L_coe.compute_CoE_Mag()
            coe_output_A = output_L_coe.compute_CoE_Ang()

            # if coe_output_A[1] > 0.16 and False: # or True only V and False only L
            #     select = 0
            #     local_seeing_count += 1
            # else:
            #     select = 1
            #     local_reasoning_count += 1

            self.Coe.append((coe_output_M[1], coe_output_A[1], select))
            # print(f"\nStep {self.step_count}: Rank: {self.accelerator.process_index}:Sample {i} CoE Mag: {coe_output_M[1]:.4f}, CoE Ang: {coe_output_A[1]:.4f}, Select: {select}")

        layer_hidden_state = None  # 释放内存

        # # ==========================================
        # # 核心修改 1: 在多卡之间同步计数 (All-Reduce)
        # # ==========================================
        # # 将计数转换为 Tensor 放入当前设备
        # counts_tensor = torch.tensor([local_seeing_count, local_reasoning_count], device=self.accelerator.device)
        
        # # 汇总所有 GPU 的计数 (sum)
        # # 这样 global_seeing_count 就是整个 batch (所有卡) 的总和
        # global_counts = self.accelerator.reduce(counts_tensor, reduction="sum")
        
        # global_seeing_count = global_counts[0].item()
        # global_reasoning_count = global_counts[1].item()

        # # ==========================================
        # # 核心修改 2: 只在主进程打印
        # # ==========================================
        # if self.accelerator.is_main_process:
        #     print(f"\nStep{self.step_count}: Global Seeing: {global_seeing_count}, Global Reasoning: {global_reasoning_count}")


        # # ==========================================
        # # 核心修改 3: 互斥的冻结/解冻逻辑 (防止全部被冻结)
        # # ==========================================
        # if global_seeing_count <= global_reasoning_count:
        #     # 决策: 冻结 Vision，训练 Language
        #     self.freeze_vision_flag = True
        #     self.batch_result.append((global_seeing_count, global_reasoning_count, 1))
        #     if self.accelerator.is_main_process:
        #         print(f"Decision: Freeze Vision, Train Language")
        # else:
        #     # 决策: 冻结 Language，训练 Vision
        #     self.freeze_vision_flag = False
        #     self.batch_result.append((global_seeing_count, global_reasoning_count, 0))
        #     if self.accelerator.is_main_process:
        #         print(f"Decision: Freeze Language, Train Vision")



        if self.step_count != 0 and self.step_count % 50 == 0: # 50
            coe = self.Coe
            batch_result = self.batch_result

            save_root = os.path.join(
                self.args.output_dir.replace("output", "coe_train_result"),
                "coe_select_results",
                "step_checkpoint"
            )
            os.makedirs(save_root, exist_ok=True)

            rank = self.accelerator.process_index
            step = self.step_count

            # ==================================================
            # 1️⃣ 当前 step 保存
            # ==================================================
            coe_path = os.path.join(
                save_root, f"Step{step}_Rank{rank}_coe.pkl"
            )
            batch_path = os.path.join(
                save_root, f"Step{step}_Rank{rank}_batch_result.pkl"
            )

            with open(coe_path, "wb") as f:
                pickle.dump(coe, f)

            # with open(batch_path, "wb") as f:
            #     pickle.dump(batch_result, f)

            # ==================================================
            # 2️⃣ 删除往前数第二次保存（step - 100）
            # ==================================================
            prev2_step = step - 100 # 100
            if prev2_step > 0:
                old_coe_path = os.path.join(
                    save_root, f"Step{prev2_step}_Rank{rank}_coe.pkl"
                )
                old_batch_path = os.path.join(
                    save_root, f"Step{prev2_step}_Rank{rank}_batch_result.pkl"
                )

                if os.path.exists(old_coe_path):
                    os.remove(old_coe_path)

                # if os.path.exists(old_batch_path):
                #     os.remove(old_batch_path)


        return (loss, outputs) if return_outputs else loss

   