```{eval-rst}
.. index:: 
    single: TensorFlow 2; convert BERT Transformer Models
```


# Converting TensorFlow 2 BERT Transformer Models

The following examples demonstrate converting TensorFlow 2 models to Core ML using Core ML Tools.

## Convert the DistilBERT Transformer Model

The following example converts the [DistilBERT model from Huggingface](https://huggingface.co/transformers/model_doc/distilbert.html#tfdistilbertformaskedlm) to Core ML. 

```{admonition} Install Transformers

You may need to first install Transformers version 4.17.0.
```

Follow these steps:

1. Add the import statements:
    
	```python
	import numpy as np
	import coremltools as ct
	import tensorflow as tf

	from transformers import DistilBertTokenizer, TFDistilBertForMaskedLM
	```

2. Load the DistilBERT model and tokenizer. This example uses the `TFDistilBertForMaskedLM` variant:
    
	```python
	tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
	distilbert_model = TFDistilBertForMaskedLM.from_pretrained('distilbert-base-cased')
	```

3. Describe and set the input layer, and then build the TensorFlow model (`tf_model`):
	
	```python
	max_seq_length = 10
	input_shape = (1, max_seq_length) #(batch_size, maximum_sequence_length)

	input_layer = tf.keras.layers.Input(shape=input_shape[1:], dtype=tf.int32, name='input')

	prediction_model = distilbert_model(input_layer)
	tf_model = tf.keras.models.Model(inputs=input_layer, outputs=prediction_model)
	```

4. Convert the `tf_model` to a Core ML neural network (`mlmodel`):
	
	```python
	mlmodel = ct.convert(tf_model)
	```

5. Create the input using `tokenizer`:
	
	```python
	# Fill the input with zeros to adhere to input_shape
	input_values = np.zeros(input_shape)
	# Store the tokens from our sample sentence into the input
	input_values[0,:8] = np.array(tokenizer.encode("Hello, my dog is cute")).astype(np.int32)
	```

6. Use `mlmodel` for prediction:
	
	```python
	mlmodel.predict({'input':input_values}) # 'input' is the name of our input layer from (3)
	```

## Convert the TF Hub BERT Transformer Model

The following example converts the [BERT model from TensorFlow Hub](https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1). Follow these steps:

1. Add the import statements:
	
	```python
	import numpy as np
	import tensorflow as tf
	import tensorflow_hub as tf_hub

	import coremltools as ct
	```

2. Describe and set the input layer:
	
	```python
	max_seq_length = 384
	input_shape = (1, max_seq_length)

	input_words = tf.keras.layers.Input(
		shape=input_shape[1:], dtype=tf.int32, name='input_words')
	input_masks = tf.keras.layers.Input(
		shape=input_shape[1:], dtype=tf.int32, name='input_masks')
	segment_ids = tf.keras.layers.Input(
		shape=input_shape[1:], dtype=tf.int32, name='segment_ids')
	```

3. Build the TensorFlow model (`tf_model`):
	
	```python
	bert_layer = tf_hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1", trainable=False)

	pooled_output, sequence_output = bert_layer(
		[input_words, input_masks, segment_ids])

	tf_model = tf.keras.models.Model(
		inputs=[input_words, input_masks, segment_ids],
		outputs=[pooled_output, sequence_output])
	```

4. Convert the `tf_model` to a neural network:
	
	```python
	mlmodel = ct.convert(tf_model, source='TensorFlow')
	```



