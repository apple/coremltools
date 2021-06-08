#ifndef __AUDIOFEATUREPRINT_ENUMS_H
#define __AUDIOFEATUREPRINT_ENUMS_H
enum MLAudioFeaturePrintAudioFeaturePrintType: int {
    MLAudioFeaturePrintAudioFeaturePrintType_sound = 20,
    MLAudioFeaturePrintAudioFeaturePrintType_NOT_SET = 0,
};

__attribute__((__unused__))
static const char * MLAudioFeaturePrintAudioFeaturePrintType_Name(MLAudioFeaturePrintAudioFeaturePrintType x) {
    switch (x) {
        case MLAudioFeaturePrintAudioFeaturePrintType_sound:
            return "MLAudioFeaturePrintAudioFeaturePrintType_sound";
        case MLAudioFeaturePrintAudioFeaturePrintType_NOT_SET:
            return "INVALID";
    }
    return "INVALID";
}

enum MLSoundVersion: int {
    MLSoundVersionSOUND_VERSION_INVALID = 0,
    MLSoundVersionSOUND_VERSION_1 = 1,
};

#endif
