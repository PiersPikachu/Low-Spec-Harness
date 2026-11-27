#!/bin/bash
# Installs SearXNG in this WSL distro (run as root by searxng_setup.bat). Re-running updates it.
# Layout as in the SearXNG docs (docs.searxng.org/admin/installation-searxng.html): user 'searxng',
# source /usr/local/searxng/searxng-src, venv /usr/local/searxng/searx-pyenv, settings /etc/searxng/settings.yml.
# Listens on 127.0.0.1:8888 only (reachable from Windows through WSL's localhost forwarding, not from the network).
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
step() { echo; echo "=== $*"; }

step "packages"
apt-get update -q
apt-get install -y -q python3-dev python3-babel python3-venv python-is-python3 git build-essential \
    libxslt-dev zlib1g-dev libffi-dev libssl-dev curl

step "user and folders"
id searxng >/dev/null 2>&1 || useradd --shell /bin/bash --system --home-dir /usr/local/searxng \
    --comment 'Privacy-respecting metasearch engine' searxng
mkdir -p /usr/local/searxng
chown -R searxng:searxng /usr/local/searxng

step "source"
if [ -d /usr/local/searxng/searxng-src/.git ]; then
    sudo -H -u searxng git -C /usr/local/searxng/searxng-src pull --ff-only
else
    sudo -H -u searxng git clone --depth 1 https://github.com/searxng/searxng /usr/local/searxng/searxng-src
fi

step "python environment"
sudo -H -u searxng bash -c '
set -e
[ -x /usr/local/searxng/searx-pyenv/bin/python ] || python3 -m venv /usr/local/searxng/searx-pyenv
. /usr/local/searxng/searx-pyenv/bin/activate
pip install -q -U pip setuptools wheel pyyaml msgspec typing-extensions pybind11
cd /usr/local/searxng/searxng-src
pip install -q --use-pep517 --no-build-isolation -e .
'

step "settings"
mkdir -p /etc/searxng
if [ ! -f /etc/searxng/settings.yml ]; then
    key=$(head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n')
    cat > /etc/searxng/settings.yml <<YML
# SearXNG for cube-harness (written by tools/searxng/setup.sh; edit freely - setup keeps an existing file)
use_default_settings: true
general:
  instance_name: "cube-harness search"
server:
  secret_key: "$key"
  bind_address: "127.0.0.1"
  port: 8888
  limiter: false
  public_instance: false
  image_proxy: false
search:
  safe_search: 1
  formats:
    - html
    - json
YML
fi
chown searxng:searxng /etc/searxng/settings.yml
chmod 600 /etc/searxng/settings.yml

step "start script"
cat > /usr/local/searxng/run.sh <<'RUN'
#!/bin/bash
# Started by the web plugin (or searxng_start.bat). One instance only.
if curl -s -o /dev/null "http://127.0.0.1:8888/healthz"; then exit 0; fi
cd /usr/local/searxng/searxng-src
. /usr/local/searxng/searx-pyenv/bin/activate
export SEARXNG_SETTINGS_PATH=/etc/searxng/settings.yml
exec python -m searx.webapp
RUN
chmod 755 /usr/local/searxng/run.sh
chown searxng:searxng /usr/local/searxng/run.sh

step "test"
sudo -H -u searxng bash -c 'nohup /usr/local/searxng/run.sh >/tmp/searxng.log 2>&1 &'
for i in $(seq 1 60); do
    if curl -s -o /dev/null "http://127.0.0.1:8888/healthz"; then break; fi
    sleep 1
done
code=$(curl -s -o /tmp/searxng_test.json -w '%{http_code}' "http://127.0.0.1:8888/search?q=rust+programming+language&format=json" || true)
n=$(python3 -c "import json;print(len(json.load(open('/tmp/searxng_test.json')).get('results',[])))" 2>/dev/null || echo 0)
echo "JSON search inside WSL: HTTP $code, $n results"
if [ "$code" != "200" ] || [ "$n" = "0" ]; then
    echo "--- last lines of the SearXNG log:"; tail -n 30 /tmp/searxng.log || true
    exit 1
fi
echo "SEARXNG SETUP OK"
