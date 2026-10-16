"""Create lossless PNG requests without a Python imaging dependency."""

import struct
import zlib


def encode_png(pixels, width, height, channels=3, bit_depth=8):
    color_type = {1: 0, 3: 2, 4: 6}[channels]
    stride = width * channels * (bit_depth // 8)
    assert len(pixels) == stride * height

    def chunk(kind, content):
        return (struct.pack(">I", len(content)) + kind + content +
                struct.pack(">I", zlib.crc32(kind + content)))

    scanlines = b"".join(b"\0" + pixels[row * stride:(row + 1) * stride] for row in range(height))
    header = struct.pack(">IIBBBBB", width, height, bit_depth, color_type, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) +
            chunk(b"IDAT", zlib.compress(scanlines)) + chunk(b"IEND", b""))
