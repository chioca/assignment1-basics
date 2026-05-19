import torch
from torch import nn
import wandb
from cs336_basics.tokenizer.tokenizer import Tokenizer
import numpy as np
from cs336_basics.train.config import TrainingConfig, ModelConfig
from cs336_basics.train.utils import (
    get_ctx,
    BatchState,
    data_loading_sequential,
    cross_entropy,
    gradient_clipping,
    get_lr_cosine_schedule,
    save_checkpoint,
)
from cs336_basics.utils import print_color
import os


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_config: TrainingConfig,
):

    # Load training dataset
    print_color(train_config.train_data_path, "red")
    original_data = np.memmap(
        train_config.train_data_path,
        dtype=np.uint16,
        mode="r+",
    )
    x = torch.from_numpy(original_data)

    best_eval_loss = float("inf")
    ctx = get_ctx(train_config.use_mixed_precision, train_config.device)

    # Training loop
    state = BatchState(pos=0)
    for step in range(train_config.num_steps):
        log_dict = {}

        inputs, targets = data_loading_sequential(
            data=x,
            batch_size=train_config.batch_size,
            context_length=model.max_seq_len,
            device=train_config.device,
            state=state,
        )

        # Forward pass
        with ctx:
            logits, aux = model(inputs)

            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
            loss = cross_entropy(logits, targets)

            if model.config.use_moe:
                # Scale z-loss
                z_loss_scaled = aux["z_loss_scaled"]
                moe_layers = aux["moe_layers"]
                loss = loss + (z_loss_scaled / moe_layers)

                lb_loss = aux["lb_loss_scaled"]
                loss = loss + (lb_loss / moe_layers)

        # Backward pass and optimization step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Gradient clipping
        gradient_clipping(model.parameters(), max_l2_norm=train_config.max_grad_norm)

        # Learning rate scheduling
        lr = get_lr_cosine_schedule(
            t=step,
            a_max=train_config.max_lr,
            a_min=train_config.min_lr,
            t_w=train_config.warmup_steps,
            t_c=train_config.num_steps - train_config.warmup_steps,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.step()

        # Logging
        if train_config.wandb_logging:
            log_dict["train/loss"] = loss.item()
            log_dict["train/perplexity"] = torch.exp(loss).item()
            log_dict["train/lr"] = lr

        print(
            f"Step {step + 1}/{train_config.num_steps}, Loss: {loss.item():.4f}, LR: {lr:.6f}",
            "green",
        )
        if model.config.use_moe:
            tokens_per_expert = aux["tokens_per_expert"]
            if model.config.use_moe and (step % train_config.log_moe_every == 0):
                layers_to_log = sorted(
                    set([0, model.config.num_layers // 2, model.config.num_layers - 1])
                )
                for layer_idx in layers_to_log:
                    tpe = (
                        tokens_per_expert[layer_idx].detach().float().cpu().numpy()
                    )  # (E,)
                    msg = " | ".join([f"E{e}:{tpe[e]:.3f}" for e in range(len(tpe))])
                    print(
                        f"[step {step}] Layer {layer_idx} tokens_per_expert: {msg}",
                        "magenta",
                    )
                    if train_config.wandb_logging:
                        for e in range(len(tpe)):
                            log_dict[f"moe/layer_{layer_idx}_expert_{e}_tokens"] = tpe[
                                e
                            ]

        # if (
        #     train_config.eval_log_interval > 0
        #     and (step + 1) % train_config.eval_log_interval == 0
        # ):
        #     # Cleanup
        #     del inputs, targets, logits, loss
        #     clear_memory()

        #     print("Evaluating model...", "blue")
        #     eval_loss, eval_perplexity = eval_model(model, train_config)
        #     if train_config.wandb_logging:
        #         log_dict["eval/loss"] = eval_loss.item()
        #         log_dict["eval/perplexity"] = eval_perplexity.item()

        #     print_color(
        #         f"Eval Loss: {eval_loss.item():.4f}, Eval Perplexity: {eval_perplexity.item():.4f}",
        #         "blue",
        #     )
        #     if eval_loss < best_eval_loss:
        #         best_eval_loss = eval_loss
        #         print_color(f"New best eval loss: {best_eval_loss:.4f}", "yellow")
        #         out_path = os.path.join(
        #             train_config.save_checkpoint_dir,
        #             train_config.model_name,
        #             f"best_model_step_{step + 1}.pt",
        #         )
        #         save_checkpoint(
        #             model=model,
        #             optimizer=optimizer,
        #             iteration=step + 1,
        #             out=out_path,
        #             verbose=True,
        #         )

        # # Sample generation
        # if (
        #     train_config.sampling_log_interval > 0
        #     and (step + 1) % train_config.sampling_log_interval == 0
        # ):
        #     generated_outputs = generate(
        #         model=model,
        #         prompt="Once upon a time",
        #         tokenizer=tokenizer,
        #         max_new_tokens=256,
        #         top_k=50,
        #         temperature=0.8,
        #     )
        #     generated_text = generated_outputs["generated_text"]
        #     print_color(f"Generated text at step {step + 1}:", "cyan")
        #     print("Once upon a time", end="")
        #     print_color(f"{generated_text}\n", "cyan")

        if train_config.wandb_logging and log_dict:
            wandb.log(log_dict, step=step + 1)
