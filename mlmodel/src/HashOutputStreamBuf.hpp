#ifndef ML_HASH_OUTPUT_STREAM_BUF_HPP
#define ML_HASH_OUTPUT_STREAM_BUF_HPP

#include <streambuf>
#include <vector>

namespace CoreML {
    namespace detail {
        struct HashOutputStreamBufImpl;
    }

    /// HashOutputStreamBuf is an output std::streambuf which calculates a hash of the stream as
    /// std::ostream writes the data.
    ///
    /// The data stream is discarded immediately.
    class HashOutputStreamBuf : public std::streambuf {
    private:
        std::shared_ptr<detail::HashOutputStreamBufImpl> m_impl;
    public:
        HashOutputStreamBuf();

        /// Returns a hash of the written stream.
        ///
        /// The client should not push any more data after calling this method.
        std::vector<std::uint8_t> hash() const;

    protected:
        std::streamsize xsputn(const char_type* s, std::streamsize n) override;
        int overflow(int ch) override;
    };
}

#endif

