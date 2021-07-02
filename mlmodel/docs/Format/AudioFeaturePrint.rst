AudioFeaturePrint
=========================


Specific audio feature print types.
   
Sound extracts features useful for identifying the predominant sound in
the audio signal.

VERSION_1 is available on iOS,tvOS 15.0+, macOS 12.0+. It uses a variable-length
input audio sample vector and yields a 512 float feature vector.

.. code-block:: proto

	/**
	* A model which takes an input audio and outputs array(s) of features
	* according to the specified feature types
	*/
	message AudioFeaturePrint {

		// Specific audio feature print types
   
		// Sound extracts features useful for identifying the predominant
		// sound in audio signal
		message Sound {
			enum SoundVersion {
				SOUND_VERSION_INVALID = 0;
				// VERSION_1 is available on iOS,tvOS 15.0+, macOS 12.0+
				// It uses a variable-length input audio sample vector and yields a 512 float feature vector
				SOUND_VERSION_1 = 1;
			}
		
			SoundVersion version = 1;
		}

		// Audio feature print type
		oneof AudioFeaturePrintType {
			Sound sound = 20;
		}
	}


