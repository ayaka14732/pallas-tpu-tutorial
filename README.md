# Pallas TPU Kernel 开发教程

## 查看教程

浏览器直接访问 <https://ayaka14732.github.io/pallas-tpu-tutorial/>。

## 本地预览文档

```bash
cd docs
pip install -r requirements.txt
sphinx-autobuild source build/html
```

然后打开 <http://127.0.0.1:8000/>。修改文档源文件后页面会自动重新构建并刷新。

如果只想手动构建一次：

```bash
cd docs
sphinx-build -b html source build/html
```

构建结果在 `docs/build/html/index.html`。

## 参考资源

- [JAX Pallas 官方文档](https://docs.jax.dev/en/latest/pallas/index.html)
- [JAX GitHub 仓库](https://github.com/jax-ml/jax)（Pallas TPU tests 和 production kernels）
- [Ragged Paged Attention 源码](https://github.com/jax-ml/jax/tree/main/jax/experimental/pallas/ops/tpu/ragged_paged_attention)
