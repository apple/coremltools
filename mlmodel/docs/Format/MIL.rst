MILspec.Program
=========================

CoreML.Specification.MILSpec
----------------------------

The top-level container. Programs, functions, blocks, ops, and tensor types all
can contain an optional set of attributes.

Identifiers, generally used for names and keys, must match the
regular expression ``[A-Za-z\_][A-Za-z0-9\_@]*``.


	
Program
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A Program is a container with following information:

	- Set of functions.
	- Each function defines a program block to be executed.
	- A model can have multiple functions defined and will have a single point
	  of entry.

Requirements:

- Must be unique within the containing program.
- Names must be valid identifiers as described above.
- Any other attributes not described by other fields.
- Keys must be valid identifiers as described above.

.. code-block:: proto

	message Program {
		int64 version = 1;

		// Must be unique within the containing program
		// Names must be valid identifiers as described above.
		map<string, Function> functions = 2;

		string docString = 3;

		// Any other attributes not described by other fields.
		// Keys must be valid identifiers as described above.
		map<string, Value> attributes = 4;
	}



Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A program-level function. A function consists of:

	- List of named inputs and output types.
	- A block defining scope for a function -- similar to a function in C/C++.

Function inputs are unordered ``(name, ValueType)`` pairs.
Inputs intended to process images must be rank-4 Float32 tensors. Dimensions
are interpreted as ``NCHW``, with ``N == 1`` and ``C`` being ``1`` for grayscale 
and ``3`` for RGB. Names must be valid identifiers as described above.

The active block is drawn from this named specialization.
This key must exist in ``block_specializations``.

Specialization keys are the name of the opset that the
function specialization is written in. They must be valid
identifiers as described above.

Outputs from all blocks must match. They define the outputs
of the function.
Each block inherits the lexical scope from the function.

Any other attributes not described by other fields.
Keys must be valid identifiers as described above.

.. code-block:: proto

	// A program-level function.
	message Function {

		// Function inputs are unordered (name, ValueType) pairs.
		// Inputs intended to process images must be rank-4 Float32 tensors. Dimensions
		// are interpreted as NCHW, with N == 1 and C being 1 for grayscale and 3 for RGB.
		// Names must be valid identifiers as described above.
		repeated NamedValueType inputs = 1;

		// The active block is drawn from this named specialization.
		// This key must exist in `block_specializations`.
		string opset = 2;

		// Named specializations of this function.
		//
		// Specialization keys are the name of the opset that the
		// function specialization is written in. They must be valid
		// identifiers as described above.
		//
		// Outputs from all blocks must match. They define the outputs
		// of the function.
		// Each block inherits the lexical scope from the function.
		map<string, Block> block_specializations = 3;

		// Any other attributes not described by other fields.
		// Keys must be valid identifiers as described above.
		map<string, Value> attributes = 4;
	}


Block
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A Block consists of:

- List of named inputs and output names
- Topologically sorted Ops

Infrequently used, these are for operators that may need to give
block-local names to input values (e.g. while_loop).

The names to give to values returned by this block. They must be
identifiers as described above.

ValueType of ``outputs[i]`` is ``Operation[j].outputs[k].type`` where 
``i``, ``j`` and ``k`` are indices of block output, block Operation, and
operation ``j`` output respectively. This is due to:

1. An operation can have more than one output.
2. Any one of operation's output could be potentially block's output.

Any other attributes not described by other fields.
Keys must be valid identifiers as described above.

.. code-block:: proto

	// A basic block with a single entry and exit in SSA form.
	message Block {
		// Infrequently used, these are for operators that may need to give
		// block-local names to input values (e.g. while_loop).
		repeated NamedValueType inputs = 1;

		// The names to give to values returned by this block. They must be
		// identifiers as described above.
		//
		// ValueType of outputs[i] is Operation[j].outputs[k].type where 
		// i, j and k are indices of block output, block Operation and
		// jth operation's output respectively.
		// this is due to
		// 1. An operation can have more than one output
		// 2. Any one of operation's output could be potentially block's output
		repeated string outputs = 2;

		repeated Operation operations = 3;

		// Any other attributes not described by other fields.
		// Keys must be valid identifiers as described above.
		map<string, Value> attributes = 4;
	}


Argument
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Argument is list of Binding to either name or value.

.. code-block:: proto

	// Argument is list of Binding to either name or value
	message Argument {
		message Binding {
			oneof binding {
				// The name of a previously defined value.
				string name = 1;

				// A compile time constant.
				Value value = 2;
			}
		}

		repeated Binding arguments = 1;
	};



Op
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A single operation/node/layer.

An Op consists of:

- List of named inputs and outputs (name, type) pair
- Optionally, blocks for Control-Flow

Operator arguments:

- Key: parameter name
- Value: Argument (list of bindings). Value is list of argument binding to
  given parameter. Binding can be a string name (previous operation output
  or input given to model/block/function), or a Value (known compile time
  value for given operation).
- Argument can be of length 1 (general) or variable length (for example, a
  concat layer).

For example:

	| ``{'stride' : ['input_01']}``
	| ``{'x' : ['input_01', 'input_02', 'input_03', false]}``

.. code-block:: proto

	// A single operation/node/layer.
	message Operation {
		// Examples: "convolution", "cropResize". Operation type defines the
		// expected inputs and output.
		string type = 1;

		// Operator arguments
		//
		// Key: parameter name
		// Value: Argument (list of bindings)
		//
		// Value is list of argument binding to given parameter
		// Binding can be a string name (previous operation output or input given to model/block/function)
		//             or a Value (known compile time value for given operation)
		// Argument can be of length 1 (general) or variable length (e.g. concat layer)
		// e.g. {'stride' : ['input_01']}
		// e.g. {'x' : ['input_01', 'input_02', 'input_03', false]}
		map<string, Argument> inputs = 2;

		// Names to which to bind values returned by this operation.
		// Names must be:
		//  (*) valid identifiers as described above; and
		//  (*) unique within the current scope.
		repeated NamedValueType outputs = 3;

		// Nested blocks for loops and conditionals. For example,
		// a conditional block will have two entries here.
		repeated Block blocks = 4;

		// Any other information not captured by other fields.
		// Keys must be valid identifiers as described above.
		map<string, Value> attributes = 5;
	}


NamedValueType
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The name of this parameter; must be a valid identifier as described above.

.. code-block:: proto

	// Named Value parameters
	// (name, type) pair
	message NamedValueType {
		// The name of this parameter; must be a valid identifier as described above.
		string name = 1;

		// This parameter's required type.
		ValueType type = 2;
	}

 
Types
-----

Primer: Two fundamental representations of state:

Variable: Variables are *never* materialized at compile time and are only
available at run time. Therefore, for Variables we only have ValueType,
which may have unknown shapes in the IR. Variable encompasses familiar
concepts such as placeholder, output of an Op.

Value: Values are ALWAYS materialized at compile time, and MAY be modified
at runtime (e.g., during on-device training). Value describes notions
such as parameter, attributes of an op. Value is either stored inside
proto (e.g., attributes) or outside of proto (e.g. parameters) and
NEVER contains unknown shape in the IR.

Comment(daviddai): A Variable with the potential to be materialized at
compile time (e.g., through constant propagation) does *not* preclude it to
be a Variable. Certain Ops such as LoadParameter and Const, their output
has potential to be materialized at compile time but is still represented
as Variable.


ValueType
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A type of any kind.

.. code-block:: proto

	message ValueType {
		oneof type {
			TensorType tensorType = 1;
			ListType listType = 2;
			TupleType tupleType = 3;
			DictionaryType dictionaryType = 4;
		}
	}



DataType
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Two schemes of specifying field id: just start with 0
without reserving numbers, but keep track of the next field ID. The
other is assign blocks of ID to int / float / uint etc.

.. code-block:: proto

	// Supported data types
	enum DataType {
		// Comment: Two schemes of specifying field id: just start with 0
		// without reserving numbers, but keep track of the next field ID. The
		// other is assign blocks of ID to int / float / uint etc.

		// 0-10 reserved for special types
		UNUSED_TYPE = 0;  // not currently in use
		BOOL = 1;
		STRING = 2;  // arbitrary sequence of bytes

		// Floats
		FLOAT16 = 10;
		FLOAT32 = 11;
		FLOAT64 = 12;

		// Ints
		INT8 = 21;
		INT16 = 22;
		INT32 = 23;
		INT64 = 24;

		// UInts
		UINT8 = 31;
		UINT16 = 32;
		UINT32 = 33;
		UINT64 = 34;
	}


TensorType
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: proto

	message TensorType {
		// The data type stored in a tensor of this type
		DataType dataType = 1;

		// The number of dimensions in the tensor shape. rank == -1 implies
		// variable (not fixed) rank
		int64 rank = 2;

		// Tensor shape values; must be of length "rank"
		repeated Dimension dimensions = 3;

		// Any other tensor type attributes not described by other fields.
		// Keys must be valid identifiers in MIL text syntax.
		map<string, Value> attributes = 4;
	}


TupleType
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: proto

	message TupleType {
		// Recursively define TupleType from ValueType.
		repeated ValueType types = 1;
	}


ListType
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: proto

	message ListType {
		// The type of element stored in a list of this type
		ValueType type = 1;

		// The number of elements in a list of this type. May be unknown (variable length)
		Dimension length = 2;
	}


DictionaryType
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: proto

	// An unordered key-value mapping
	message DictionaryType {
		ValueType keyType = 1;
		ValueType valueType = 2;
	}



Dimension
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: proto
    
	message Dimension {
		oneof dimension {
		  ConstantDimension constant = 1;
		  UnknownDimension unknown = 2;
		}

		message ConstantDimension {
			uint64 size = 1;
		}

		message UnknownDimension {
			bool variadic = 1;
		}
	}


Values
------

See the primer on variables and values at the beginning of the Types section.


Value
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: proto
    
	message Value {
		string docString = 1; // optional human-readable texts.
		ValueType type = 2;

		// An immediate value stored within the proto
		message ImmediateValue {
			oneof value {
				TensorValue tensor = 1;
				TupleValue tuple = 2;
				ListValue list = 3;
				DictionaryValue dictionary = 4;
			}
		}

		// Reference to a "blob v2" storage file
		message BlobFileValue {
			// name of file
			string fileName = 1;

			// byte offset to metadata
			uint64 offset = 2;
		}

		oneof value {
			ImmediateValue immediateValue = 3;
			BlobFileValue blobFileValue = 5;
		}
	}


TensorValue
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: proto
    
	message TensorValue {
		oneof value {
			RepeatedFloats floats = 1;
			RepeatedInts ints = 2;
			RepeatedBools bools = 3;
			RepeatedStrings strings = 4;
			RepeatedLongInts longInts = 5;
			RepeatedDoubles doubles = 6;
			RepeatedBytes bytes = 7;
		}

		message RepeatedFloats {
			repeated float values = 1 [packed = true];
		}
	
		message RepeatedDoubles {
			repeated double values = 1 [packed = true];
		}

		message RepeatedInts {
			repeated int32 values = 1 [packed = true];
		}

		message RepeatedLongInts {
			repeated int64 values = 1 [packed = true];
		}

		message RepeatedBools {
			repeated bool values = 1 [packed = true];
		}

		message RepeatedStrings {
			repeated string values = 1;
		}

		message RepeatedBytes {
			bytes values = 1;
		}
	}


TupleValue
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: proto

	message TupleValue {
		// Comment: TupleValue is recursively defined from Value.
		repeated Value values = 1;
	}


ListValue
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: proto

	message ListValue {
		repeated Value values = 1;
	}


DictionaryValue
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: proto

	message DictionaryValue {
		message KeyValuePair {
			Value key = 1;
			Value value = 2;
		}
		repeated KeyValuePair values = 1;
	}


