# Single-runner lock for long sweeps on Windows + Git Bash.  Source, then call
# acquire_lock LOCKFILE.
#
# Git Bash's $$ is an MSYS PID, but tasklist only knows Windows PIDs -- the
# first version of this lock stored $$, so every live runner looked stale.  The
# lock now records the Windows PID from /proc/$$/winpid, and is honoured only
# while that PID is alive *and* is a bash.exe, so neither a crashed runner nor
# Windows reusing its PID for an unrelated process can block a restart.

# acquire_lock LOCKFILE -> 0 and records our Windows PID, or 1 if a live runner holds it
acquire_lock() {
  local lock="$1" me other
  me=$(cat "/proc/$$/winpid" 2>/dev/null)
  if [ -z "$me" ]; then
    echo "lock: cannot determine this shell's Windows PID"
    return 1
  fi
  if [ -f "$lock" ]; then
    other=$(cat "$lock" 2>/dev/null)
    if [ -n "$other" ] && [ "$other" != "$me" ] && \
       tasklist //FI "PID eq $other" //FI "IMAGENAME eq bash.exe" //NH 2>/dev/null \
         | grep -qw "$other"; then
      echo "lock: held by live runner (Windows PID $other)"
      return 1
    fi
    echo "lock: clearing stale lock from PID $other"
  fi
  echo "$me" > "$lock"
  return 0
}
