```{eval-rst}
.. index:: 
    single: Xcode; model preview types
    single: model preview types in Xcode
```

# Xcode Model Preview Types

After converting models to the Core ML format, you can set up the Xcode preview of certain models by adding preview metadata and parameters.

## Overview

The following table shows the types of models that work with the Xcode preview feature, and the preview metadata and parameters you need to provide.

```{note}

Some model architecture types, such as Neural Network Classifier, don't require a `model.preview.type`, and some model preview types don't require preview parameters.
```

| Architecture/Preview type                                                      | model.preview.type | model.preview.parameters                                 | Input      | Output       |
| :----------------------------------------------------------------------------- | :----------------- | :------------------------------------------------------- | :--------- | :----------- |
| [Segmentation](#segmentation-example)                                          | `imageSegmenter`   | `{'labels': ['LABEL', ...], 'colors': ['HEXCODE', ...]}` | Image      | MultiArray   |
| [BERT QA](#bert-qa-example)                                                    | `bertQA`           |                                                          | MultiArray | MultiArray   |
| [Pose Estimation](#body-pose-example)                                          | `poseEstimation`   | `{"width_multiplier": FLOAT, "output_stride": INT}`      | Image      | MultiArray   |
| [Image Classifier](https://coremltools.readme.io/docs/introductory-quickstart) |                    |                                                          | Image      | Dict, string |
| Depth Estimation                                                               | `depthEstimation`  |                                                          | Image      | MultiArray   |

## Segmentation Example

The following example demonstrates how to add an Xcode preview for a [segmentation model](https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/ "DeepLabV3 model with a ResNet-101 backbone"). Follow these steps:

1. Load the converted model.
2. Set up the parameters. This example collects them in `labels_json`.
3. Define the `model.preview.type` metadata as `"imageSegmenter"`.
4. Define the `model.preview.parameters` as `labels_json`.
5. Save the model.

```python
# load the model
mlmodel = ct.models.MLModel("SegmentationModel_no_metadata.mlmodel")

labels_json = {"labels": ["background", "aeroplane", "bicycle", "bird", "board", "bottle", "bus", "car", "cat", "chair", "cow", "diningTable", "dog", "horse", "motorbike", "person", "pottedPlant", "sheep", "sofa", "train", "tvOrMonitor"]}

mlmodel.user_defined_metadata["com.apple.coreml.model.preview.type"] = "imageSegmenter"
mlmodel.user_defined_metadata['com.apple.coreml.model.preview.params'] = json.dumps(labels_json)

mlmodel.save("SegmentationModel_with_metadata.mlmodel")
```

```{note}

For the full code to convert the model, see [Converting a PyTorch Segmentation Model](pytorch-conversion-examples).
```

### Open the Model in Xcode

To launch Xcode and open the model information pane, double-click the saved `SegmentationModel_with_metadata.mlmodel` file in the Mac Finder.

![SegmentationModel_with_metadata.mlmodel](images/xcode-segment-metadata3.png)

The Segmentation model for this example offers tabs for **Metadata**, **Preview**, **Predictions**, and **Utilities**. Click the **Predictions** tab to see the model’s input and output.

![Predictions tab](images/xcode-quickstart6-model-predictions2.png)


### Preview the Model in Xcode

To preview the model’s output for a given input, follow these steps:

```{note}

The preview for a segmentation model is available in Xcode 12.3 or newer.
```

1. Click the **Preview** tab.
2. Drag an image into the image well on the left side of the model preview. The result appears in the preview pane:

![Preview pane](images/xcode-deeplab-model5-preview-drag.png)
    
![Cropped preview](images/xcode-segment-preview2-crop.png)

```{note}

For the full code to convert the model, see [Converting a PyTorch Segmentation Model](pytorch-conversion-examples).
```

## BERT QA Example

For a [BERT QA model](https://developer.apple.com/machine-learning/models/#text), follow these steps to add the metadata for the Xcode preview:

1. Load the converted model.
2. Define the `model.preview.type` metadata as `"bertqa"`.
3. Save the model.

```python
model = ct.models.MLModel("BERT_no_preview_type.mlmodel")
model.user_defined_metadata["com.apple.coreml.model.preview.type"] = "bertQA"
model.save("BERT_with_preview_type.mlmodel")
```

### Open the Model in Xcode

Double-click the `BERT_with_preview_type.mlmodel` file in the Mac Finder to launch Xcode and open the model information pane:

![Model information page](images/xcode_bert_model_preview1.png)

The BERT QA model for this example offers tabs for **Metadata**, **Preview**, **Predictions**, and **Utilities**. Click the **Predictions** tab to see the model’s input and output.

![Predictions tab](images/xcode_bert_model_preview2_prediction.png)


### Preview the Model in Xcode

To preview the model’s performance, follow these steps:

1. Click the **Preview** tab.
2. Copy and paste sample text, such as the BERT QA model description, into the Passage Context field.
3. Enter a question in the Question field, such as **What is BERT?** The answer appears in the Answer Candidate field, and is also highlighted in the Passage Context field.

![Preview performance](images/xcode_bert_model_preview3_preview.png)


## Body Pose Example

For a body pose model such as [PoseNet](https://developer.apple.com/documentation/coreml/detecting_human_body_poses_in_an_image?language=objc), follow these steps to add the metadata for an Xcode preview:

1. Load the converted model.
2. Define the `model.preview.type` metadata as `"poseEstimation"`.
3. Provide the preview parameters for `{"width_multiplier": FLOAT, "output_stride": INT}`. You can learn more about these parameters in [posenet_model.ts in Pre-trained TensorFlow.js models on GitHub](https://github.com/tensorflow/tfjs-models/blob/master/posenet/src/posenet_model.ts "tfjs-models/posenet/src/posenet_model.ts").
4. Save the model.

```python
model = ct.models.MLModel("posenet_no_preview_type.mlmodel") 
model.user_defined_metadata["com.apple.coreml.model.preview.type"] = "poseEstimation"
params_json = {"width_multiplier": 1.0, "output_stride": 16}
model.user_defined_metadata['com.apple.coreml.model.preview.params'] = json.dumps(params_json)
model.save("posenet_with_preview_type.mlmodel")
```

### Open the Model in Xcode

Double-click the `posenet_with_preview_type.mlmodel` file in the Mac Finder to launch Xcode and open the model information pane:

![Model information pane for PoseNet model](images/xcode-bodypose-posenet-metadata.png)

The PoseNet model for this example offers tabs for **Metadata**, **Preview**, **Predictions**, and **Utilities**. Click the **Predictions** tab to see the model’s input and output.

![PoseNet model input and output](images/xcode-bodypose-posenet-predictions.png)


### Preview the Model in Xcode

To preview the model's output for a given input, follow these steps using the following sample image:

```{figure} images/standing-man.jpeg
:alt: Standing man (standing-man.jpeg)
:align: center
:class: imgnoborder

Right-click and choose **Save Image** to download this test image. ("Figure from a Crèche: Standing Man" is in the public domain, available from [creativecommons.org](https://creativecommons.org).)
```

1. Click the **Preview** tab.
2. Drag the above image into the image well on the left side of the model preview.
3. The result appears in the preview pane:

![Preview of result](images/xcode-bodypose-posenet-preview.png)

The result shows a single pose estimation (the key points of the body pose) under the **Single** tab, and estimates of multiple poses under the **Multiple** tab.

