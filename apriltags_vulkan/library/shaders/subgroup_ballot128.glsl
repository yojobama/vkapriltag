// Helpers for treating a subgroupBallot() result as one flat bitmask wider
// than 32 bits. subgroupBallot() returns a uvec4 (up to 128 lanes: bit i of
// component i/32), and this device's subgroup size can exceed 32 - the RX
// 9060 XT test machine reports subgroupSize=64, so a ballot's set bits can
// legitimately span both .x and .y. Reading only .x (as an earlier version
// of this file's callers did) silently drops any lane numbered 32 or above,
// which is exactly the kind of bug that only shows up on hardware wider
// than 32 lanes - Mali-G610's subgroup size of 16 never exercised it.
// Included only by shaders that already declare
// GL_KHR_shader_subgroup_ballot (this doesn't declare it itself, since
// declaring an unused extension in a file with no other subgroup calls
// would be a needless requirement).

// Index of the lowest set bit across all 128 bits, or 128 if `v` is zero
// (never dereferenced by the callers here, which only call this after
// checking AnyBits128(v)).
uint FindLSB128(uvec4 v) {
  if (v.x != 0u) return uint(findLSB(v.x));
  if (v.y != 0u) return 32u + uint(findLSB(v.y));
  if (v.z != 0u) return 64u + uint(findLSB(v.z));
  if (v.w != 0u) return 96u + uint(findLSB(v.w));
  return 128u;
}

uint PopCount128(uvec4 v) {
  return uint(bitCount(v.x)) + uint(bitCount(v.y)) + uint(bitCount(v.z)) + uint(bitCount(v.w));
}

bool AnyBits128(uvec4 v) {
  return (v.x | v.y | v.z | v.w) != 0u;
}

uvec4 AndNot128(uvec4 a, uvec4 b) {
  return a & ~b;
}
