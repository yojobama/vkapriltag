"""Detects the tags of a corpus frame through the SWIG Python module and checks the accessors and the errors."""
import struct
import sys
import zlib

module_dir, corpus = sys.argv[1], sys.argv[2]
sys.path.insert(0, module_dir)
import vkapriltag as vk  # noqa: E402


def read_grey_png(path):
    """An 8-bit greyscale, non-interlaced PNG as (width, height, bytes)."""
    data = open(path, "rb").read()
    pos, idat, width = 8, b"", 0
    while pos < len(data):
        n, kind = struct.unpack(">I4s", data[pos:pos + 8])
        body = data[pos + 8:pos + 8 + n]
        if kind == b"IHDR":
            width, height, depth, colour, _, _, interlace = struct.unpack(">IIBBBBB", body)
            assert depth == 8 and colour == 0 and interlace == 0
        elif kind == b"IDAT":
            idat += body
        pos += 12 + n
    raw = zlib.decompress(idat)
    out, prev = bytearray(), bytearray(width)
    for y in range(height):
        f = raw[y * (width + 1)]
        row = bytearray(raw[y * (width + 1) + 1:(y + 1) * (width + 1)])
        for x in range(width):
            a = row[x - 1] if x else 0
            b = prev[x]
            c = prev[x - 1] if x else 0
            if f == 1:
                row[x] = (row[x] + a) & 255
            elif f == 2:
                row[x] = (row[x] + b) & 255
            elif f == 3:
                row[x] = (row[x] + (a + b) // 2) & 255
            elif f == 4:
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pred = a if pa <= pb and pa <= pc else (b if pb <= pc else c)
                row[x] = (row[x] + pred) & 255
        out += row
        prev = row
    return width, height, bytes(out)


width, height, frame = read_grey_png(corpus + "/one-tag/one-tag_640x480_000.png")
options = vk.DetectorOptions()
options.width, options.height = width, height
detector = vk.Detector(options)
assert detector.width() == width and detector.height() == height

result = detector.DetectBuffer(frame)
assert result.ready and len(result.detections) == 1, len(result.detections)
tag = result.detections[0]
assert tag.id == 316 and tag.family_name() == "tag36h11" and tag.hamming == 0
assert 0 < tag.centre_x() < width and 0 < tag.centre_y() < height
xs = [tag.corner_x(i) for i in range(4)]
assert min(xs) < tag.centre_x() < max(xs)
assert any(tag.homography(i) != 0.0 for i in range(9))
assert result.timing.total_ms > 0 and result.timing.candidates >= 1

# The result is a copy: it survives the next call.
again = detector.DetectBuffer(bytearray(frame))
assert again.detections[0].id == 316 and tag.id == 316

# Other input forms and the stride.
assert detector.DetectBytes(vk.ByteVector(list(frame))).detections[0].id == 316
padded = b"".join(frame[y * width:(y + 1) * width] + bytes(8) for y in range(height))
assert detector.DetectBuffer(padded, width + 8).detections[0].id == 316

try:
    import numpy as np
    arr = np.frombuffer(frame, dtype=np.uint8).reshape(height, width)
    assert detector.DetectBuffer(arr).detections[0].id == 316
except ImportError:
    pass

# Pose and options.
o = vk.DetectorOptions()
o.width, o.height = width, height
o.intrinsics.fx = o.intrinsics.fy = 600.0
o.intrinsics.cx, o.intrinsics.cy = 320.0, 240.0
o.tag_size = 0.1
o.families = vk.StringVector(["tag36h11"])
posed = vk.Detector(o).DetectBuffer(frame)
assert len(posed.poses) == 1 and posed.poses[0].valid and posed.poses[0].translation(2) > 0

# Pipelined mode through the same module.
o.mode = vk.Mode_kPipelined
pipelined = vk.Detector(o)
assert not pipelined.DetectBuffer(frame).ready
assert pipelined.Flush().ready

# Errors become Python exceptions.
for call, kind in (
    (lambda: detector.DetectBuffer(frame[:100]), ValueError),
    (lambda: vk.Detector(vk.DetectorOptions()), ValueError),
):
    try:
        call()
    except kind:
        pass
    else:
        raise AssertionError("expected " + kind.__name__)
bad = vk.DetectorOptions()
bad.width, bad.height = width, height
bad.families = vk.StringVector(["tagNope"])
try:
    vk.Detector(bad)
except ValueError as e:
    assert "tagNope" in str(e)
else:
    raise AssertionError("expected ValueError")
print("ok")
