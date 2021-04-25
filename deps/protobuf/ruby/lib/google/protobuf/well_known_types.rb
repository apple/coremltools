#!/usr/bin/ruby
# Protocol Buffers - Google's data interchange format
# Copyright 2008 Google Inc.  All rights reserved.
# https://developers.google.com/protocol-buffers/
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#     * Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above
# copyright notice, this list of conditions and the following disclaimer
# in the documentation and/or other materials provided with the
# distribution.
#     * Neither the name of Google Inc. nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

require 'google/protobuf/any_pb'
require 'google/protobuf/duration_pb'
require 'google/protobuf/field_mask_pb'
require 'google/protobuf/struct_pb'
require 'google/protobuf/timestamp_pb'

module Google
  module Protobuf
    Any.class_eval do
      def pack(msg, type_url_prefix = 'type.googleapis.com/')
        self.type_url = if type_url_prefix.empty? || (type_url_prefix[-1] != '/')
                          "#{type_url_prefix}/#{msg.class.descriptor.name}"
                        else
                          "#{type_url_prefix}#{msg.class.descriptor.name}"
                        end
        self.value = msg.to_proto
      end

      def unpack(klass)
        klass.decode(value) if is(klass)
      end

      def type_name
        type_url.split('/')[-1]
      end

      def is(klass)
        type_name == klass.descriptor.name
      end
    end

    Timestamp.class_eval do
      def to_time
        Time.at(to_f)
      end

      def from_time(time)
        self.seconds = time.to_i
        self.nanos = time.nsec
      end

      def to_i
        seconds
      end

      def to_f
        seconds + (nanos.to_f / 1_000_000_000)
      end
    end

    Duration.class_eval do
      def to_f
        seconds + (nanos.to_f / 1_000_000_000)
      end
    end

    class UnexpectedStructType < Google::Protobuf::Error; end

    Value.class_eval do
      def to_ruby(recursive = false)
        case kind
        when :struct_value
          if recursive
            struct_value.to_h
          else
            struct_value
          end
        when :list_value
          if recursive
            list_value.to_a
          else
            list_value
          end
        when :null_value
          nil
        when :number_value
          number_value
        when :string_value
          string_value
        when :bool_value
          bool_value
        else
          raise UnexpectedStructType
        end
      end

      def from_ruby(value)
        case value
        when NilClass
          self.null_value = 0
        when Numeric
          self.number_value = value
        when String
          self.string_value = value
        when TrueClass
          self.bool_value = true
        when FalseClass
          self.bool_value = false
        when Struct
          self.struct_value = value
        when Hash
          self.struct_value = Struct.from_hash(value)
        when ListValue
          self.list_value = value
        when Array
          self.list_value = ListValue.from_a(value)
        else
          raise UnexpectedStructType
        end
      end
    end

    Struct.class_eval do
      def [](key)
        fields[key].to_ruby
      end

      def []=(key, value)
        raise UnexpectedStructType, 'Struct keys must be strings.' unless key.is_a?(String)

        fields[key] ||= Google::Protobuf::Value.new
        fields[key].from_ruby(value)
      end

      def to_h
        ret = {}
        fields.each { |key, val| ret[key] = val.to_ruby(true) }
        ret
      end

      def self.from_hash(hash)
        ret = Struct.new
        hash.each { |key, val| ret[key] = val }
        ret
      end
    end

    ListValue.class_eval do
      include Enumerable

      def length
        values.length
      end

      def [](index)
        values[index].to_ruby
      end

      def []=(index, value)
        values[index].from_ruby(value)
      end

      def <<(value)
        wrapper = Google::Protobuf::Value.new
        wrapper.from_ruby(value)
        values << wrapper
      end

      def each
        values.each { |x| yield(x.to_ruby) }
      end

      def to_a
        values.map { |x| x.to_ruby(true) }
      end

      def self.from_a(arr)
        ret = ListValue.new
        arr.each { |val| ret << val }
        ret
      end
    end
  end
end
