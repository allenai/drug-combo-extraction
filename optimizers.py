from transformers import AdamW, get_linear_schedule_with_warmup

def adamw_with_linear_warmup(named_parameters, lr, correct_bias, num_train_optimization_steps, warmup_proportion):
        # Decay scheme taken from https://github.com/princeton-nlp/PURE/blob/main/run_relation.py#L384.
        param_optimizer = list(named_parameters)
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer
                        if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer
                        if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr, correct_bias=correct_bias)
        optimization_steps = int(num_train_optimization_steps * warmup_proportion)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    optimization_steps,
                                                    num_train_optimization_steps)
        return {
            'optimizer': optimizer,
            'scheduler': scheduler,
        }

def simple_adamw(named_parameters, lr, correct_bias, num_train_optimization_steps=None, warmup_proportion=None):
    parameters = [p for n, p in named_parameters]
    return AdamW(parameters, lr=lr, correct_bias=correct_bias)