#ifndef __WORDTAGGER_ENUMS_H
#define __WORDTAGGER_ENUMS_H
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
enum MLWordTaggerTags: int {
    MLWordTaggerTags_stringTags = 200,
    MLWordTaggerTags_NOT_SET = 0,
};

static const char * MLWordTaggerTags_Name(MLWordTaggerTags x) {
    switch (x) {
        case MLWordTaggerTags_stringTags:
            return "MLWordTaggerTags_stringTags";
        case MLWordTaggerTags_NOT_SET:
            return "INVALID";
    }
}

#pragma clang diagnostic pop
#endif
