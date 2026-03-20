#!/usr/bin/env bash
set -euxo pipefail

export DEBIAN_FRONTEND=noninteractive

journalctl --vacuum-size=200M || true
apt-get clean || true
rm -f /var/log/*.gz /var/log/*.[0-9] || true

if ! id -u mavinomichael >/dev/null 2>&1; then
  useradd -m -s /bin/bash mavinomichael
fi
usermod -aG sudo mavinomichael || true

apt-get update
apt-get install -y \
  ca-certificates \
  curl \
  git \
  git-lfs \
  htop \
  jq \
  rsync \
  tmux \
  unzip \
  wget \
  zip \
  build-essential \
  python3-pip

mkdir -p /mnt/data /mnt/agentgym
chown -R mavinomichael:mavinomichael /mnt/data /mnt/agentgym /home/mavinomichael

if ! grep -q '^vm.max_map_count=262144' /etc/sysctl.conf; then
  echo 'vm.max_map_count=262144' >> /etc/sysctl.conf
  sysctl -p || true
fi

su - mavinomichael -c 'git lfs install --skip-repo || true'
su - mavinomichael -c 'touch ~/.agentgym_startup_done'

df -h /
