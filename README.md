Initialize:
```bash
git clone https://github.com/HaochenGu/mirage_softmax.git
git submodule update --init --recursive
cd mirage_softmax
pip install -e . -v
export MIRAGE_HOME=$pwd
```

Test Softmax(precision change at top):
```bash
python test_softmax_persistent_kernel.py
```

Test Mask-Attention(Multi-Token):
```bash
cd tests/runtime_python
pip install .
python test_multitoken_decoding.py
```

Top-K Generation(Based on Qwen3):
```bash
cd demo/qwen3
python demo_tree.py
python demo_tree.py --use-mirage
```