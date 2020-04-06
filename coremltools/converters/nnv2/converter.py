from .nnv2_program.passes.common_pass import common_pass


class ConverterRegistry:
    frontends = {}
    backends = {}

    @staticmethod
    def frontend(converter):
        ConverterRegistry.frontends[converter.name] = converter
        return converter

    @staticmethod
    def backend(converter):
        ConverterRegistry.backends[converter.name] = converter
        return converter


# TODO: <rdar://problem/57081777> Docstring for converter's frontend interfaces

@ConverterRegistry.frontend
class NitroSSAConverter:
    """Deprecated and use NNv2DummyFrontend"""
    name = "nitrossa"

    def __call__(self, model, *args, **kwargs):
        return model


@ConverterRegistry.frontend
class TensorFlowFrontend:
    name = 'tensorflow'

    def __call__(self, *args, **kwargs):
        from .frontend.tensorflow import load
        return load(*args, **kwargs)


@ConverterRegistry.frontend
class TensorFlow2Frontend:
    name = 'tensorflow2'

    def __call__(self, *args, **kwargs):
        from .frontend.tensorflow2 import load
        return load(*args, **kwargs)


@ConverterRegistry.backend
class NNv1ProtoBackend:
    name = "nnv1_proto"

    def __call__(self, *args, **kwargs):
        from .backend.nnv1 import load
        return load(*args, **kwargs)


@ConverterRegistry.frontend
class CustomFrontend:
    name = "custom"

    def __call__(self, *args, **kwargs):
        from coremltools.converters.nnv2.nnv2_program.passes.common_pass import common_pass
        return common_pass(*args, **kwargs)


@ConverterRegistry.frontend
class NNv2DummyFrontend:
    "A dummy frontend that returns the identical nnv2 program"
    name = "nnv2_program"

    def __call__(self, program, *args, **kwargs):
        return program


def convert(model, convert_from='TensorFlow', convert_to='nnv1_proto',
            converter_registry=ConverterRegistry, **kwargs):
    """
    Convert from an external representation.

    Args:
        model (Any): The model to convert.
        convert_from (str): The name of the input converter.
        convert_to (str): The name of the output converter.
        converter_registry: Converter registries.
    Returns:
        The converted model.
    """
    frontend_converter_type = converter_registry.frontends.get(convert_from.lower())
    if not frontend_converter_type:
        msg = 'Frontend converter "{}" not implemented, must be one of: {}'
        raise NotImplementedError(msg.format(
            convert_from, list(converter_registry.frontends.keys())))
    frontend_converter = frontend_converter_type()

    backend_converter_type = converter_registry.backends.get(convert_to.lower())
    if not backend_converter_type:
        msg = 'Backend converter "{}" not implemented, must be one of: {}'
        raise NotImplementedError(msg.format(
            convert_to, list(converter_registry.backends.keys())))
    backend_converter = backend_converter_type()

    prog = frontend_converter(model, **kwargs)
    common_pass(prog)
    out = backend_converter(prog, **kwargs)

    return out
