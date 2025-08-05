```{eval-rst}
.. index:: 
    single: TensorFlow 1; convert DeepSpeech model
```


# Converting a TensorFlow 1 DeepSpeech Model

The following example explores the automatic handling of flexible shapes and other related capabilities of the Core ML Tools converter. It uses an [automatic speech recognition](https://en.wikipedia.org/wiki/Speech_recognition#End-to-end_automatic_speech_recognition) (ASR) task in which the input is a speech audio file and the output is the text transcription of it.

The ASR system for this example consists of three stages: preprocessing, post-processing, and a neural network model between them that does most of the heavy lifting. The preprocessing and post-processing stages employ standard techniques which can be easily implemented. The focus of this example is on converting the neural network model.

```{admonition} How ASR Works

Preprocessing involves extracting the [Mel-frequency cepstral coefficients](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum) (MFCCs) from the raw audio file. The MFCCs are fed into the neural network model, which returns a character-level time series of probability distributions. Those are then postprocessed by a CTC decoder to produce the final transcription.

The example uses a pre-trained TensorFlow model called [DeepSpeech](https://github.com/mozilla/DeepSpeech/blob/master/DeepSpeech.py) that uses [long short-term memory](https://en.wikipedia.org/wiki/Long_short-term_memory) (LSTM) and a few dense layers stacked on top of each other — an architecture common for `seq2seq` models.

```

## Set Up the Model

To run this example on your system, follow these steps:

1. Download the following assets:
    - Processing and inspection utilities ([demo_utils.py](https://docs-assets.developer.apple.com/coremltools/deepspeech/demo_utils.py)) 
    - Sample audio file ([audio_sample_16bit_mono_16khz.wav](https://docs-assets.developer.apple.com/coremltools/deepspeech/audio_sample_16bit_mono_16khz.wav))
    - Alphabet configuration file ([alphabet.txt](https://github.com/mozilla/DeepSpeech/blob/master/data/alphabet.txt))
    - Language model scorer ([kenlm.scorer](https://github.com/mozilla/DeepSpeech/blob/master/data/lm/kenlm.scorer))
    - Pre-trained weights ([deepspeech-0.7.1-checkpoint](https://github.com/mozilla/DeepSpeech/releases/download/v0.7.1/deepspeech-0.7.1-checkpoint.tar.gz))
    - Script to export TensorFlow 1 model ([DeepSpeech.py](https://github.com/mozilla/DeepSpeech/blob/master/DeepSpeech.py))

2. Install the `deepspeech` package using `pip`:
    
    ```shell
    pip install deepspeech
    ```

3. Run the following script downloaded from the [DeepSpeech repository](https://github.com/mozilla/DeepSpeech) to export the TensorFlow 1 model:
    
	```shell
	python DeepSpeech.py --export_dir /tmp --checkpoint_dir ./deepspeech-0.7.1-checkpoint --alphabet_config_path=alphabet.txt --scorer_path=kenlm.scorer >/dev/null 2>&1
	```

4. After the model is exported, inspect the outputs of the TensorFlow graph:
    
	```python
	tf_model = "/tmp/output_graph.pb"
	from demo_utils import inspect_tf_outputs
	inspect_tf_outputs(tf_model)
	```
    
    The TensorFlow graph outputs are `'mfccs'`, `'logits'`, `'new_state_c'`, and `'new_state_h'`. 
    
    The `'mfccs'` output represents the output of the preprocessing stage. This means that the exported TensorFlow graph contains not just the DeepSpeech model, but also the preprocessing subgraph. 

5. Strip off this preprocessing component by providing the remaining three output names to the unified converter function:
    
	```python
	outputs = ["logits", "new_state_c", "new_state_h"]
	```


## Convert the Model and Preprocess an Audio File

Preprocessing and post-processing functions have already been constructed using code in the [DeepSpeech repository](https://github.com/mozilla/DeepSpeech). To convert the model and preprocess an audio file, follow these steps:

1. Convert the model to a Core ML neural network model:
    
	```python
	import coremltools as ct
	mlmodel = ct.convert(tf_model, outputs=outputs)
	```

2. After the model is converted, load and preprocess an audio file:
    
	```python
	audiofile = "./audio_sample_16bit_mono_16khz.wav"
	from demo_utils import preprocessing, postprocessing
	mfccs = preprocessing(audiofile)
	print(mfccs.shape)
	```

Preprocessing transforms the audio file into a tensor object of shape `(1, 636, 19, 26)`. The shape of the tensor can be viewed as one audio file, preprocessed into 636 sequences, each of width 19, and containing 26 coefficients. The number of sequences changes with the length of the audio. In this 12-second audio file there are 636 sequences.

## Feed the Input Into the Model

Inspect the input shapes that the Core ML model expects:

```python
from demo_utils import inspect_inputs
inspect_inputs(mlmodel, tf_model)
```

The model input with the name `input_node` has the shape `(1, 16, 19, 26)` which matches the shape of the preprocessed tensor in all the dimensions except for the sequence dimension. Since the converted Core ML model can process only 16 sequences at a time, create a loop to break the input features into chunks and feed each segment into the model one-by-one:

```python
start = 0 
step = 16
max_time_steps = mfccs.shape[1]
logits_sequence = []

input_dict = {}
input_dict["input_lengths"]  = np.array([step]).astype(np.float32)
input_dict["previous_state_c"] = np.zeros([1, 2048]).astype(np.float32) # Initializing cell state 
input_dict["previous_state_h"] = np.zeros([1, 2048]).astype(np.float32) # Initializing hidden state 

print("Transcription: \n")

while (start + step) < max_time_steps:
    input_dict["input_node"] = mfccs[:, start:(start + step), :, :]

    # Evaluation
    preds = mlmodel.predict(input_dict)

    start += step
    logits_sequence.append(preds["logits"])

    # Updating states
    input_dict["previous_state_c"] = preds["new_state_c"]
    input_dict["previous_state_h"] = preds["new_state_h"]

    # Decoding
    probs = np.concatenate(logits_sequence)
    transcription = postprocessing(probs)
    print(transcription[0][1], end="\r", flush=True)
```

The above code breaks the preprocessed feature into size-16 slices, and runs a prediction on each slice, along with state management, inside a loop. After running the above code, the transcription matches the contents of the audio file.

## Use a Dynamic TensorFlow Model

It is also possible to run the prediction on the entire preprocessed feature in just one go using a dynamic TensorFlow model. Follow these steps:

1. Rerun the same script from the DeepSpeech repository to obtain a dynamic graph. Provide an additional flag `n_steps` which corresponds to the sequence length and has a default value of 16. Setting it to -1 means that the sequence length can take any positive value:
    
	```shell
	!python DeepSpeech.py --n_steps -1 --export_dir /tmp --checkpoint_dir ./deepspeech-0.7.1-checkpoint --alphabet_config_path=alphabet.txt --scorer_path=kenlm.scorer >/dev/null 2>&1
	```

2. Convert the newly exported dynamic TensorFlow model to a Core ML neural network model:
    
	```python
	mlmodel = ct.convert(tf_model, outputs=outputs)
	```

3. After the model is converted, inspect how this new model is different from the previous static one:
    
	```python
	inspect_inputs(mlmodel,tf_model)
	```

The shape of input `input_node` is now `(1, None, 19, 26)`, which mean that this CoreML model can work on inputs of arbitrary-sequence length.

```{note}

The dynamic Core ML model offers dynamic operations, such as "get shape" and "dynamic reshape", which are not available in the previous static model. The Core ML Tools converter offers the same simplicity with dynamic models as it does with static models.
```

4. Validate the transcription accuracy on the same audio file:

```python
input_dict = {}
input_dict["input_node"] = mfccs
input_dict["input_lengths"] = np.array([mfccs.shape[1]]).astype(np.float32)
input_dict["previous_state_c"] = np.zeros([1, 2048]).astype(np.float32) # Initializing cell state 
input_dict["previous_state_h"] = np.zeros([1, 2048]).astype(np.float32) # Initializing hidden state
```

5. With the dynamic model you don't need to create a loop. You can feed the entire input feature directly into the model:

```python
probs = mlmodel.predict(input_dict)["logits"]
transcription = postprocessing(probs)
print(transcription[0][1])
```

The result is the same transcription with the dynamic Core ML model as with the static model.

## Convert a Dynamic Model to a Static One

So far you worked with two variants of the DeepSpeech model:

- Static TF graph: The converter produced a Core ML neural network model with inputs of fixed shape. 
- Dynamic model: The converter produced a Core ML neural network model that can accept inputs of any sequence length. 

The converter handles both cases transparently without needing to make a change to the conversion call.

It is also possible with the Core ML Tools converter to start with a dynamic TF graph and obtain a static Core ML model. Provide the type description object containing the name and shape of the input to the conversion API: 

```python
input = ct.TensorType(name="input_node", shape=(1,16,19,26))
mlmodel = ct.convert(tf_model, outputs=outputs, inputs=[input])
```

Under the hood, the type and value inference propagates this shape information to remove all the unnecessary dynamic operations. 

Static models are likely to be more performant while the dynamic ones are more flexible.
