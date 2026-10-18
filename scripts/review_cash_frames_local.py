"""Second local model pass for visibility triage, NOT human verified ground truth."""
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_cash_sft import read, write
from scripts.train_cash_lora import batch, load_base, loss_forward


def main():
    import torch
    from transformers import AutoProcessor
    original = {e['id']: e for e in read(ROOT / 'outputs/cash-lora/examples.json')}
    selected = read(ROOT / 'outputs/cash-diagnostic/examples.json')
    processor, model = load_base(ROOT / 'models/Qwen3-VL-2B-Instruct')
    model.eval()
    with torch.no_grad():
        inputs = batch(processor, original[selected[0]['id']])
        reference_loss = float(model(**inputs, use_cache=False).loss)
        optimized_loss = float(loss_forward(model, inputs))
        write(ROOT / 'outputs/cash-diagnostic/loss_mask_reference.json', {
            'transformers_builtin_assistant_masked_loss': reference_loss,
            'optimized_loss': optimized_loss, 'absolute_difference': abs(reference_loss-optimized_loss),
            'passed': abs(reference_loss-optimized_loss) < 1e-3,
            'assistant_tokens': int((inputs['labels'] != -100).sum()),
            'prompt_masked_tokens': int((inputs['labels'] == -100).sum())})
        del inputs
    rows = []
    for e in selected:
        if e['split'] != 'train':
            continue
        for variant, source in [('original224', original[e['id']]), ('context336', e)]:
            example = copy.deepcopy(source)
            side = 224 if variant == 'original224' else 336
            processor = AutoProcessor.from_pretrained(ROOT / 'models/Qwen3-VL-2B-Instruct',
                local_files_only=True, min_pixels=32 * 32, max_pixels=side * side)
            example['prompt'] = (
                'Опиши только видимые действия по порядку кадров. '
                'Отдельно укажи: 1) передачу и обработку наличных, '
                '2) уход одного посетителя и начало взаимодействия со следующим, '
                '3) видно ли перемещение первой пачки со стола в шкаф, '
                '4) можно ли проследить ту же пачку после смены посетителя. '
                'Для каждого утверждения укажи индексы кадров. Если это неразличимо, '
                'напиши «неразличимо». Не делай юридический вывод и не додумывай '
                'действия между кадрами. Ответ до 150 слов.')
            inputs = batch(processor, example, target=False)
            with torch.inference_mode():
                tokens = model.generate(**inputs, do_sample=False, max_new_tokens=320,
                                        repetition_penalty=1.05, no_repeat_ngram_size=12)
            generated = tokens[0, inputs['input_ids'].shape[1]:]
            text = processor.decode(generated, skip_special_tokens=True)
            row = {'id': e['id'], 'variant': variant, 'model_description': text,
                   'generated_tokens': len(generated),
                   'processor_max_pixels': side * side,
                   'actual_image_grid_thw': inputs['image_grid_thw'].tolist(),
                   'truncated': len(generated) >= 320,
                   'not_ground_truth': True, 'human_visibility_verified': False}
            rows.append(row)
            write(ROOT / 'outputs/cash-diagnostic/local_visibility_triage.json', rows)
            print(e['id'], variant, 'saved; not independent ground truth', flush=True)
            del inputs, tokens


if __name__ == '__main__':
    main()
