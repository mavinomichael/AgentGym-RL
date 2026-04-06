#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=/dev/null
source "$SCRIPT_DIR/common.sh"

REPO_ROOT=$(multi_agent::repo_root)
SERVER_ROOT="$REPO_ROOT/AgentGym/agentenv-webarena"
WEB_ARENA_ENV_FILE="${WEB_ARENA_ENV_FILE:-$SERVER_ROOT/.env}"
ASSET_ROOT="${WEB_ARENA_ASSET_ROOT:-$HOME/agentgym_runs/webarena_assets}"
DOWNLOAD_ROOT="$ASSET_ROOT/downloads"
HOMEPAGE_ROOT="$SERVER_ROOT/webarena/environment_docker/webarena-homepage"
HOSTNAME_VALUE="${WEB_ARENA_HOSTNAME:-127.0.0.1}"
LOCAL_MAP_URL="http://$HOSTNAME_VALUE:3000"
PUBLIC_MAP_URL="${WEB_ARENA_PUBLIC_MAP_URL:-http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:3000/}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "$DOWNLOAD_ROOT"

if ! command -v docker >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y docker.io
  sudo systemctl enable docker
  sudo systemctl start docker
fi

docker_cmd() {
  if docker info >/dev/null 2>&1; then
    docker "$@"
  else
    sudo docker "$@"
  fi
}

download_if_missing() {
  local url="$1"
  local output="$2"
  local tmp="${output}.partial"
  if [[ -s "$output" ]]; then
    return 0
  fi
  rm -f "$tmp"
  wget "$url" -O "$tmp"
  mv "$tmp" "$output"
}

content_length_for() {
  local url="$1"
  curl -fsSI "$url" | awk 'tolower($1) == "content-length:" {gsub("\r", "", $2); print $2; exit}'
}

download_verified_asset() {
  local url="$1"
  local output="$2"
  local expected_size="${3:-}"
  local tmp="${output}.partial"
  if [[ -s "$output" ]]; then
    if [[ -z "$expected_size" ]]; then
      return 0
    fi
    if [[ "$(stat -c%s "$output")" == "$expected_size" ]]; then
      return 0
    fi
    rm -f "$output"
  fi
  rm -f "$tmp"
  wget "$url" -O "$tmp"
  if [[ -n "$expected_size" ]] && [[ "$(stat -c%s "$tmp")" != "$expected_size" ]]; then
    printf 'Downloaded asset size mismatch for %s (expected %s bytes).\n' "$output" "$expected_size" >&2
    exit 1
  fi
  mv "$tmp" "$output"
}

ensure_image_loaded_from_tar() {
  local image_ref="$1"
  local tar_path="$2"
  if ! docker_cmd image inspect "$image_ref" >/dev/null 2>&1; then
    docker_cmd load --input "$tar_path"
  fi
  rm -f "$tar_path"
}

ensure_container_running() {
  local name="$1"
  shift
  if docker_cmd ps --format '{{.Names}}' | grep -Fxq "$name"; then
    return 0
  fi
  if docker_cmd ps -a --format '{{.Names}}' | grep -Fxq "$name"; then
    docker_cmd start "$name"
  else
    docker_cmd run "$@"
  fi
}

WIKI_URL="http://metis.lti.cs.cmu.edu/webarena-images/wikipedia_en_all_maxi_2022-05.zim"
WIKI_EXPECTED_SIZE="$(content_length_for "$WIKI_URL")"
download_verified_asset "$WIKI_URL" "$DOWNLOAD_ROOT/wikipedia_en_all_maxi_2022-05.zim" "$WIKI_EXPECTED_SIZE"

download_if_missing "http://metis.lti.cs.cmu.edu/webarena-images/shopping_final_0712.tar" "$DOWNLOAD_ROOT/shopping_final_0712.tar"
ensure_image_loaded_from_tar "shopping_final_0712:latest" "$DOWNLOAD_ROOT/shopping_final_0712.tar"

download_if_missing "http://metis.lti.cs.cmu.edu/webarena-images/shopping_admin_final_0719.tar" "$DOWNLOAD_ROOT/shopping_admin_final_0719.tar"
ensure_image_loaded_from_tar "shopping_admin_final_0719:latest" "$DOWNLOAD_ROOT/shopping_admin_final_0719.tar"

download_if_missing "http://metis.lti.cs.cmu.edu/webarena-images/postmill-populated-exposed-withimg.tar" "$DOWNLOAD_ROOT/postmill-populated-exposed-withimg.tar"
ensure_image_loaded_from_tar "postmill-populated-exposed-withimg:latest" "$DOWNLOAD_ROOT/postmill-populated-exposed-withimg.tar"

download_if_missing "http://metis.lti.cs.cmu.edu/webarena-images/gitlab-populated-final-port8023.tar" "$DOWNLOAD_ROOT/gitlab-populated-final-port8023.tar"
ensure_image_loaded_from_tar "gitlab-populated-final-port8023:latest" "$DOWNLOAD_ROOT/gitlab-populated-final-port8023.tar"

ensure_container_running shopping --name shopping -p 7770:80 -d shopping_final_0712
ensure_container_running shopping_admin --name shopping_admin -p 7780:80 -d shopping_admin_final_0719
ensure_container_running forum --name forum -p 9999:80 -d postmill-populated-exposed-withimg
ensure_container_running gitlab --name gitlab -d -p 8023:8023 gitlab-populated-final-port8023 /opt/gitlab/embedded/bin/runsvdir-start
docker_cmd rm -f kiwix33 >/dev/null 2>&1 || true
docker_cmd run --name kiwix33 --volume "$DOWNLOAD_ROOT:/data" -p 8888:80 -d ghcr.io/kiwix/kiwix-serve:3.3.0 wikipedia_en_all_maxi_2022-05.zim >/dev/null

sleep 60

docker_cmd exec shopping /var/www/magento2/bin/magento setup:store-config:set --base-url="http://$HOSTNAME_VALUE:7770"
docker_cmd exec shopping mysql -u magentouser -pMyPassword magentodb -e "UPDATE core_config_data SET value='http://$HOSTNAME_VALUE:7770/' WHERE path = 'web/secure/base_url';"
docker_cmd exec shopping /var/www/magento2/bin/magento cache:flush

docker_cmd exec shopping_admin php /var/www/magento2/bin/magento config:set admin/security/password_is_forced 0
docker_cmd exec shopping_admin php /var/www/magento2/bin/magento config:set admin/security/password_lifetime 0
docker_cmd exec shopping_admin /var/www/magento2/bin/magento setup:store-config:set --base-url="http://$HOSTNAME_VALUE:7780"
docker_cmd exec shopping_admin mysql -u magentouser -pMyPassword magentodb -e "UPDATE core_config_data SET value='http://$HOSTNAME_VALUE:7780/' WHERE path = 'web/secure/base_url';"
docker_cmd exec shopping_admin /var/www/magento2/bin/magento cache:flush

docker_cmd exec gitlab sed -i "s|^external_url.*|external_url 'http://$HOSTNAME_VALUE:8023'|" /etc/gitlab/gitlab.rb
docker_cmd exec gitlab gitlab-ctl reconfigure

"$PYTHON_BIN" -m pip install flask
perl -0pi -e "s|<your-server-hostname>|http://$HOSTNAME_VALUE|g" "$HOMEPAGE_ROOT/templates/index.html"
pkill -f "python3 app.py" || true
(cd "$HOMEPAGE_ROOT" && nohup "$PYTHON_BIN" app.py >"$ASSET_ROOT/homepage.log" 2>&1 &)

export SHOPPING="http://$HOSTNAME_VALUE:7770"
export SHOPPING_ADMIN="http://$HOSTNAME_VALUE:7780/admin"
export REDDIT="http://$HOSTNAME_VALUE:9999"
export GITLAB="http://$HOSTNAME_VALUE:8023"
if [[ -z "${MAP:-}" ]]; then
  if [[ "${ALLOW_PUBLIC_MAP_FALLBACK:-0}" == "1" ]]; then
    export MAP="$PUBLIC_MAP_URL"
  else
    export MAP="$LOCAL_MAP_URL"
  fi
fi
export WIKIPEDIA="http://$HOSTNAME_VALUE:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
export HOMEPAGE="http://$HOSTNAME_VALUE:4399"

if ! curl -fsS --max-time 20 "$WIKIPEDIA" >/dev/null; then
  printf 'Local WebArena Wikipedia is not healthy at %s.\n' "$WIKIPEDIA" >&2
  exit 1
fi

multi_agent::write_webarena_env_file "$WEB_ARENA_ENV_FILE"

printf 'WebArena websites bootstrapped.\\n'
printf '  %-20s %s\\n' 'SHOPPING' "$SHOPPING"
printf '  %-20s %s\\n' 'SHOPPING_ADMIN' "$SHOPPING_ADMIN"
printf '  %-20s %s\\n' 'REDDIT' "$REDDIT"
printf '  %-20s %s\\n' 'GITLAB' "$GITLAB"
printf '  %-20s %s\\n' 'MAP' "$MAP"
printf '  %-20s %s\\n' 'WIKIPEDIA' "$WIKIPEDIA"
printf '  %-20s %s\\n' 'HOMEPAGE' "$HOMEPAGE"
if [[ "$MAP" == "$PUBLIC_MAP_URL" ]]; then
  printf 'WARNING: MAP is using the public WebArena map endpoint (%s), not a local port-3000 deployment.\\n' "$MAP"
elif [[ "$MAP" != "$LOCAL_MAP_URL" ]]; then
  printf 'WARNING: MAP is using a custom override (%s).\\n' "$MAP"
fi
