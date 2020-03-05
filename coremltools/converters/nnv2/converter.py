from ._deps import HAS_PYESPRESSO

from .nnv2_program.passes.common_pass import common_pass


class ConverterRegistry:
    frontends = {}
    backends = {}

    @staticmethod
    def frontend(converter):
        ConverterRegistry.frontends[converter.NAME] = converter
        return converter

    @staticmethod
    def backend(converter):
        ConverterRegistry.backends[converter.NAME] = converter
        return converter


# TODO: <rdar://problem/57081777> Docstring for converter's frontend interfaces

@ConverterRegistry.frontend
class NitroSSAConverter:
    NAME = "NitroSSA"

    def __call__(self, model, *args, **kwargs):
        return model

@ConverterRegistry.frontend
class TensorflowFrontend:
    NAME = "tensorflow"

    def __call__(self, *args, **kwargs):
        from .frontend.tensorflow import load
        return load(*args, **kwargs)

@ConverterRegistry.backend
class NNv1ProtoBackend:
    NAME = "nnv1_proto"

    def __call__(self, *args, **kwargs):
        from .backend.nnv1 import load as backend_load
        return backend_load(*args, **kwargs)

if HAS_PYESPRESSO:

    @ConverterRegistry.backend
    class NespressoBackend:
        NAME = "nespresso"

        def __call__(self, *args, **kwargs):
            raise NotImplemented("Nespresso backend is not implemented")
            from .backend.nespresso import load as backend_load
            return backend_load(*args, **kwargs)


@ConverterRegistry.frontend
class CustomFrontend:
    NAME = "custom"

    def __call__(self, *args, **kwargs):
        from coremltools.converters.nnv2.nnv2_program.passes.common_pass import common_pass
        return common_pass(*args, **kwargs)


@ConverterRegistry.frontend
class NNv2DummyFrontend:
    "A dummy frontend that returns the identical nnv2 program"
    NAME = "nnv2_program"

    def __call__(self, program, *args, **kwargs):
        return program


def convert(model, convert_from='tensorflow', convert_to='proto', ConverterRegistry=ConverterRegistry, **kwargs):
    """
    Convert from an external representation.

    Args:
        model (Any): The model to convert.
        convert_from (str): The name of the input converter.
        convert_to (str): The name of the output converter.
    Returns:
        The converted model.
    """
    frontend_conv_type = ConverterRegistry.frontends.get(convert_from)
    if not frontend_conv_type:
        raise NotImplementedError(
            'Frontend converter "{}" not implemented'.format(convert_from))
    frontend_conv = frontend_conv_type()

    backend_conv_type = ConverterRegistry.backends.get(convert_to)
    if not backend_conv_type:
        raise NotImplementedError(
            'Backend converter "{}" not implemented'.format(convert_to))
    backend_conv = backend_conv_type()

    ir = frontend_conv(model, **kwargs)
    common_pass(ir)
    out = backend_conv(ir, **kwargs)

    return out
