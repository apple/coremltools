#include "HashOutputStreamBuf.hpp"
#include <CommonCrypto/CommonCrypto.h>

namespace CoreML {
    namespace detail {
        struct HashOutputStreamBufImpl {
            CC_SHA256_CTX context;
            std::uint8_t hash[256 / 8];
            bool finalized;
        };
    }

    HashOutputStreamBuf::HashOutputStreamBuf()
        :m_impl(new detail::HashOutputStreamBufImpl) {
        m_impl->finalized = false;
        CC_SHA256_Init(&m_impl->context);
    }

    std::streamsize HashOutputStreamBuf::xsputn(const char_type* s, std::streamsize n) {
        if (m_impl->finalized) {
            return 0;  // hash has been finalized; accept no more data.
        }
        CC_SHA256_Update(&m_impl->context, s, static_cast<CC_LONG>(n));
        return n;
    }

    int HashOutputStreamBuf::overflow(int ch) {
        if (m_impl->finalized) {
            return 0;  // hash has been finalized; accept no more data.
        }
        CC_SHA256_Update(&m_impl->context, &ch, 1);
        return 1;
    }

    std::vector<std::uint8_t> HashOutputStreamBuf::hash() const {
        if (!m_impl->finalized) {
            CC_SHA256_Final(std::const_pointer_cast<detail::HashOutputStreamBufImpl>(m_impl)->hash, &m_impl->context);
            m_impl->finalized = true;
        }
        return std::vector<std::uint8_t>(m_impl->hash, m_impl->hash + sizeof(m_impl->hash) / sizeof(m_impl->hash[0]));
    }
}

