# 自动modular转modeling

modular要转化为完全的modeling才行

```bash
conda activate transformer
cd transformers
mv xxx src/transformers/models/xxx # xxx(modular_xxx.py, configuration_xxx.py) # github文件夹下的源码而不是包
python utils/modular_model_converter.py --files_to_parse src/transformers/models/xxx/modular_xxx.py 
cp src/transformers/models/hjxa_qwen2/modeling_hjxa_qwen2.py xxx
```

# 注册

在/ruilab/jxhe/CoE_Monitor/ms-swift/swift/model/models/llm.py中注册