# Host names for sync.sh, remote.sh and detach.sh (sourced, not run). The setup is two machines
# reached over ssh: a login node, which shares ~/flybrain-doom with the GPU node over NFS, and the
# GPU node, which is reached through the login node. Both are ssh aliases, not hard-coded here:
#   FLYBRAIN_LOGIN_HOST  ssh alias of the login node, as seen from this machine   (default login-host)
#   FLYBRAIN_GPU_HOST    host name of the GPU node, as seen from the login node    (default gpu-host)
# Set them in the environment, or in scripts/hosts.local.sh (gitignored), for example:
#   : "${FLYBRAIN_LOGIN_HOST:=my-login-alias}"
#   : "${FLYBRAIN_GPU_HOST:=my-gpu-node}"
# The scripts take "login" or "gpu" as the node argument (any other value is used as a host name).
_hosts_local="$(dirname "${BASH_SOURCE[0]}")/hosts.local.sh"
if [ -f "$_hosts_local" ]; then
  # shellcheck source=/dev/null
  . "$_hosts_local"
fi
LOGIN_HOST="${FLYBRAIN_LOGIN_HOST:-login-host}"
GPU_HOST="${FLYBRAIN_GPU_HOST:-gpu-host}"

# resolve_node login|gpu|<host> -> the host name to ssh to
resolve_node() {
  case "$1" in
    login) printf '%s\n' "$LOGIN_HOST" ;;
    gpu)   printf '%s\n' "$GPU_HOST" ;;
    *)     printf '%s\n' "$1" ;;
  esac
}
