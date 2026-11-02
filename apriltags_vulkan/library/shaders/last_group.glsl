// Runs a step once, after every workgroup of the dispatch has finished its work, in place of a
// following one-workgroup dispatch. The includer defines LAST_GROUP_TICKET, a zeroed uint in a buffer,
// and calls LastGroupDone() from every invocation after its work, outside any divergent branch.
shared uint last_group_ticket;

bool LastGroupDone() {
  memoryBarrierBuffer();
  barrier();
  if (gl_LocalInvocationIndex == 0u) {
    last_group_ticket = atomicAdd(LAST_GROUP_TICKET, 1u);
  }
  barrier();
  return last_group_ticket + 1u == gl_NumWorkGroups.x * gl_NumWorkGroups.y * gl_NumWorkGroups.z;
}
