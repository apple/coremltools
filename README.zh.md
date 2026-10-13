[![Build Status](https://img.shields.io/gitlab/pipeline/coremltools1/coremltools/main)](https://gitlab.com/coremltools1/coremltools/-/pipelines?page=1&scope=branches&ref=main)
[![PyPI Release](https://img.shields.io/pypi/v/coremltools.svg)](#)
[![Python Versions](https://img.shields.io/pypi/pyversions/coremltools.svg)](#)

[Core ML Tools](https://apple.github.io/coremltools/docs-guides/source/overview-coremltools.html)
=======================

<p>
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

![Core ML Tools logo](docs/logo.png)

使用 [Core ML Tools](https://apple.github.io/coremltools/docs-guides/source/overview-coremltools.html)（*coremltools*）将来自第三方框架的机器学习模型转换为 Core ML 格式。该 Python 工具包提供了对以下主流训练框架的模型转换支持：

* [TensorFlow 1.x](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf)
* [TensorFlow 2.x](https://www.tensorflow.org/api_docs)
* [PyTorch](https://pytorch.org/)
* 非神经网络框架：
	* [scikit-learn](https://scikit-learn.org/stable/)
	* [XGBoost](https://xgboost.readthedocs.io/en/latest/)
	* [LibSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/)

借助 coremltools，您可以：

* 将训练好的模型转换为 Core ML 格式。
* 读取、写入并优化 Core ML 模型。
* 在 macOS 上使用 Core ML 执行推理预测，以验证模型转换或构建结果。

模型转换完成后，您可以使用 Xcode 将 Core ML 模型无缝集成到您的应用程序中。

## 安装指南

最新稳定版本可通过 https://pypi.org/project/coremltools/ 获取。

使用 `pip` 或您偏好的 Python 包管理器即可安装：

```shell
pip install coremltools
```

如需安装最新开发版（dev release）：
```shell
pip install coremltools==9.1.dev1
```

## 关于 Core ML

[Core ML](https://developer.apple.com/documentation/coreml) 是 Apple 官方推出的机器学习框架，旨在将机器学习模型集成到您的应用中。Core ML 为所有类型的模型提供了统一的表示形式。您的应用可以利用 Core ML API 和用户数据完全在用户设备端进行推理预测与模型微调。Core ML 充分调动 CPU、GPU 以及神经网络引擎（Neural Engine）协同加速，在最大化端侧性能的同时，将内存占用和功耗降至最低。模型完全在用户设备本地运行，无需任何网络连接，既能切实保障用户数据隐私，又能确保应用响应迅速流畅。

## 更多资源

关于如何安装 coremltools，请参阅[安装 Core ML Tools](https://apple.github.io/coremltools/docs-guides/source/installing-coremltools.html)。如需了解更多详细信息，请参阅以下资源：

* [版本发布说明 (Release Notes)](https://github.com/apple/coremltools/releases/)
* [使用指南与示例 (Guide and examples)](https://apple.github.io/coremltools/docs-guides/index.html)
* [API 参考文档 (API Reference)](https://apple.github.io/coremltools/index.html)
* [Core ML 模型规范 (Core ML Specification)](https://apple.github.io/coremltools/mlmodel/index.html)
* [从源码构建 (Building from Source)](BUILDING.md)
* [贡献指南 (Contribution Guidelines)](CONTRIBUTING.md)

---
*注：本文档由社区志愿者维护（贡献者：[@JasonYeYuhe](https://github.com/JasonYeYuhe)），内容与英文官方文档保持同步（更新时间：2026年10月13日）。如有任何疑问或更新建议，欢迎提交 PR 或 Issue。*
