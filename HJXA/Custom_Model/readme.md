# 自动modular转modeling

```bash
conda activate transformer
cd transformers
mv xxx src/transformers/models/xxx # xxx(modular_xxx.py, configuration_xxx.py)
python utils/modular_model_converter.py --files_to_parse src/transformers/models/xxx/modular_xxx.py 
cp src/transformers/models/hjxa_qwen2/modeling_hjxa_qwen2.py 
```