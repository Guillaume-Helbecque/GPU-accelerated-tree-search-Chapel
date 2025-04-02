module util
{
  param BUSY: bool = false;
  param IDLE: bool = true;

  // Take a boolean array and return false if it contains at least one "true", "true" if not
  private inline proc _allIdle(const arr: [] atomic bool): bool
  {
    for elt in arr {
      if (elt.read() == BUSY) then return false;
    }

    return true;
  }

  proc allIdle(const arr: [] atomic bool, flag: atomic bool): bool
  {
    if flag.read() {
      return true; // fast exit
    }
    else {
      if _allIdle(arr) {
        flag.write(true);
        return true;
      }
      else {
        return false;
      }
    }
  }

  proc common_help_message(): void
  {
    writeln("\n    usage:  main.out [parameter value] ...");
    writeln("\n  General Parameters:\n");
    writeln("   --m              int   minimum number of elements to offload on a GPU device");
    writeln("   --M              int   maximum number of elements to offload on a GPU device");
    writeln("   --D              int   number of GPU device(s)");
    writeln("   --help (or -h)         this message");
  }
}
