#!/bin/bash
# Sets up the WSL distro 'cube-sandbox' for running artifact code (run as root by sandbox_setup.bat; re-run to update).
# Isolation (/etc/wsl.conf, applied after 'wsl --terminate cube-sandbox'):
#   - no Windows drives mounted (no /mnt/c ...): code in here cannot read or change your Windows files
#   - no Windows interop, its .exe handler removed and the Windows driver-store share unmounted: code in here cannot
#     start Windows programs
#   - runs as the unprivileged user 'runner'
# Not isolated: the network (code can reach the internet, like any program), and it shares the WSL VM's CPU / memory.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
step() { echo; echo "=== $*"; }

step "packages (Python, C/C++, Node.js, Go, Java, Rust, Graphviz)"
apt-get update -q
apt-get install -y -q --no-install-recommends python3 python3-pip python3-venv python-is-python3 python3-pytest \
    build-essential nodejs npm golang-go default-jdk-headless cargo graphviz ca-certificates curl

step "user"
id runner >/dev/null 2>&1 || useradd --create-home --shell /bin/bash runner
echo "user runner: $(id runner)"

step "isolation (/etc/wsl.conf)"
cat > /etc/wsl.conf <<'CONF'
# cube-sandbox: artifact code runs here, cut off from Windows (written by tools/sandbox/setup.sh)
[automount]
enabled = false
mountFsTab = false

[interop]
enabled = false
appendWindowsPath = false

[user]
default = runner

[boot]
# at every start, as root: drop the handler that hands Windows .exe files to Windows, and the read-only share of the
# Windows driver store (/usr/lib/wsl/drivers holds ~120 .exe files; only GPU code needs it, and the sandbox runs none)
command = "echo -1 > /proc/sys/fs/binfmt_misc/WSLInterop 2>/dev/null; umount -l /usr/lib/wsl/drivers 2>/dev/null; rmdir /mnt/[a-z] 2>/dev/null; true"
CONF
# the drive folders from the first start (when drives were still mounted) stay behind empty: remove them
rmdir /mnt/[a-z] 2>/dev/null || true
cat /etc/wsl.conf

step "versions"
python3 --version; gcc --version | head -n 1; node --version; go version; java -version 2>&1 | head -n 1; cargo --version; dot -V 2>&1
echo "SANDBOX PACKAGES OK"
