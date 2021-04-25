#!/usr/bin/env ruby
#
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

require 'conformance_pb'
require 'google/protobuf/test_messages_proto3_pb'

$test_count = 0
$verbose = false

def do_test(request)
  test_message = ProtobufTestMessages::Proto3::TestAllTypes.new
  response = Conformance::ConformanceResponse.new

  begin
    case request.payload
    when :protobuf_payload
      begin
        test_message = ProtobufTestMessages::Proto3::TestAllTypes.decode(
          request.protobuf_payload
        )
      rescue Google::Protobuf::ParseError => e
        response.parse_error = e.message.encode('utf-8')
        return response
      end

    when :json_payload
      begin
        test_message = ProtobufTestMessages::Proto3::TestAllTypes.decode_json(
          request.json_payload
        )
      rescue Google::Protobuf::ParseError => e
        response.parse_error = e.message.encode('utf-8')
        return response
      end

    when nil
      raise "Request didn't have payload"
    end

    case request.requested_output_format
    when :UNSPECIFIED
      raise 'Unspecified output format'

    when :PROTOBUF
      response.protobuf_payload = test_message.to_proto

    when :JSON
      response.json_payload = test_message.to_json

    when nil
      raise "Request didn't have requested output format"
    end
  rescue StandardError => e
    response.runtime_error = e.message.encode('utf-8')
  end

  response
end

# Returns true if the test ran successfully, false on legitimate EOF.
# If EOF is encountered in an unexpected place, raises IOError.
def do_test_io
  length_bytes = STDIN.read(4)
  return false if length_bytes.nil?

  length = length_bytes.unpack1('V')
  serialized_request = STDIN.read(length)
  raise IOError if serialized_request.nil? || serialized_request.length != length

  request = Conformance::ConformanceRequest.decode(serialized_request)

  response = do_test(request)

  serialized_response = Conformance::ConformanceResponse.encode(response)
  STDOUT.write([serialized_response.length].pack('V'))
  STDOUT.write(serialized_response)
  STDOUT.flush

  if $verbose
    warn("conformance_ruby: request=#{request.to_json}, " \
                                 "response=#{response.to_json}\n")
  end

  $test_count += 1

  true
end

loop do
  next if do_test_io

  warn('conformance_ruby: received EOF from test runner ' \
              "after #{$test_count} tests, exiting")
  break
end
