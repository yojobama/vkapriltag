// Helpers treating a subgroupBallot() uvec4 as one flat 128-bit mask (bit i in
// component i/32). Include only from shaders that declare
// GL_KHR_shader_subgroup_ballot.
// Index of the lowest set bit, or 128 if v is zero.
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
