# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: pb/runner.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from pb import models_pb2 as pb_dot_models__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='pb/runner.proto',
  package='bigscity.hmptoolkit.runner',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x0fpb/runner.proto\x12\x1a\x62igscity.hmptoolkit.runner\x1a\x0fpb/models.proto\"\x88\x01\n\x0cRunnerConfig\x12\x34\n\x04type\x18\x01 \x01(\x0e\x32&.bigscity.hmptoolkit.runner.RunnerType\x12\x42\n\x08\x64\x65\x65pmove\x18\x02 \x01(\x0b\x32\x30.bigscity.hmptoolkit.runner.DeepmoveRunnerConfig\"\xb0\x02\n\x14\x44\x65\x65pmoveRunnerConfig\x12\x10\n\x08use_cuda\x18\x01 \x01(\x08\x12\n\n\x02lr\x18\x02 \x01(\x02\x12\n\n\x02L2\x18\x03 \x01(\x02\x12\x11\n\tmax_epoch\x18\x04 \x01(\x05\x12\x0f\n\x07lr_step\x18\x05 \x01(\x05\x12\x10\n\x08lr_decay\x18\x06 \x01(\x02\x12\x0c\n\x04\x63lip\x18\x07 \x01(\x02\x12\x1a\n\x12schedule_threshold\x18\x08 \x01(\x02\x12\x0f\n\x07verbose\x18\t \x01(\x05\x12\x39\n\nmodel_type\x18\n \x01(\x0e\x32%.bigscity.hmptoolkit.models.ModelType\x12\x42\n\x0emodel_deepmove\x18\x0b \x01(\x0b\x32*.bigscity.hmptoolkit.models.DeepmoveConfig*1\n\nRunnerType\x12\x15\n\x11\x45MPTY_RUNNER_TYPE\x10\x00\x12\x0c\n\x08\x44\x45\x45PMOVE\x10\x01\x62\x06proto3'
  ,
  dependencies=[pb_dot_models__pb2.DESCRIPTOR,])

_RUNNERTYPE = _descriptor.EnumDescriptor(
  name='RunnerType',
  full_name='bigscity.hmptoolkit.runner.RunnerType',
  filename=None,
  file=DESCRIPTOR,
  create_key=_descriptor._internal_create_key,
  values=[
    _descriptor.EnumValueDescriptor(
      name='EMPTY_RUNNER_TYPE', index=0, number=0,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='DEEPMOVE', index=1, number=1,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=510,
  serialized_end=559,
)
_sym_db.RegisterEnumDescriptor(_RUNNERTYPE)

RunnerType = enum_type_wrapper.EnumTypeWrapper(_RUNNERTYPE)
EMPTY_RUNNER_TYPE = 0
DEEPMOVE = 1



_RUNNERCONFIG = _descriptor.Descriptor(
  name='RunnerConfig',
  full_name='bigscity.hmptoolkit.runner.RunnerConfig',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='type', full_name='bigscity.hmptoolkit.runner.RunnerConfig.type', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='deepmove', full_name='bigscity.hmptoolkit.runner.RunnerConfig.deepmove', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=65,
  serialized_end=201,
)


_DEEPMOVERUNNERCONFIG = _descriptor.Descriptor(
  name='DeepmoveRunnerConfig',
  full_name='bigscity.hmptoolkit.runner.DeepmoveRunnerConfig',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='use_cuda', full_name='bigscity.hmptoolkit.runner.DeepmoveRunnerConfig.use_cuda', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='lr', full_name='bigscity.hmptoolkit.runner.DeepmoveRunnerConfig.lr', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='L2', full_name='bigscity.hmptoolkit.runner.DeepmoveRunnerConfig.L2', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='max_epoch', full_name='bigscity.hmptoolkit.runner.DeepmoveRunnerConfig.max_epoch', index=3,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='lr_step', full_name='bigscity.hmptoolkit.runner.DeepmoveRunnerConfig.lr_step', index=4,
      number=5, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='lr_decay', full_name='bigscity.hmptoolkit.runner.DeepmoveRunnerConfig.lr_decay', index=5,
      number=6, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='clip', full_name='bigscity.hmptoolkit.runner.DeepmoveRunnerConfig.clip', index=6,
      number=7, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='schedule_threshold', full_name='bigscity.hmptoolkit.runner.DeepmoveRunnerConfig.schedule_threshold', index=7,
      number=8, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='verbose', full_name='bigscity.hmptoolkit.runner.DeepmoveRunnerConfig.verbose', index=8,
      number=9, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='model_type', full_name='bigscity.hmptoolkit.runner.DeepmoveRunnerConfig.model_type', index=9,
      number=10, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='model_deepmove', full_name='bigscity.hmptoolkit.runner.DeepmoveRunnerConfig.model_deepmove', index=10,
      number=11, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=204,
  serialized_end=508,
)

_RUNNERCONFIG.fields_by_name['type'].enum_type = _RUNNERTYPE
_RUNNERCONFIG.fields_by_name['deepmove'].message_type = _DEEPMOVERUNNERCONFIG
_DEEPMOVERUNNERCONFIG.fields_by_name['model_type'].enum_type = pb_dot_models__pb2._MODELTYPE
_DEEPMOVERUNNERCONFIG.fields_by_name['model_deepmove'].message_type = pb_dot_models__pb2._DEEPMOVECONFIG
DESCRIPTOR.message_types_by_name['RunnerConfig'] = _RUNNERCONFIG
DESCRIPTOR.message_types_by_name['DeepmoveRunnerConfig'] = _DEEPMOVERUNNERCONFIG
DESCRIPTOR.enum_types_by_name['RunnerType'] = _RUNNERTYPE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

RunnerConfig = _reflection.GeneratedProtocolMessageType('RunnerConfig', (_message.Message,), {
  'DESCRIPTOR' : _RUNNERCONFIG,
  '__module__' : 'pb.runner_pb2'
  # @@protoc_insertion_point(class_scope:bigscity.hmptoolkit.runner.RunnerConfig)
  })
_sym_db.RegisterMessage(RunnerConfig)

DeepmoveRunnerConfig = _reflection.GeneratedProtocolMessageType('DeepmoveRunnerConfig', (_message.Message,), {
  'DESCRIPTOR' : _DEEPMOVERUNNERCONFIG,
  '__module__' : 'pb.runner_pb2'
  # @@protoc_insertion_point(class_scope:bigscity.hmptoolkit.runner.DeepmoveRunnerConfig)
  })
_sym_db.RegisterMessage(DeepmoveRunnerConfig)


# @@protoc_insertion_point(module_scope)
