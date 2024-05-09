#! /usr/bin/env bash

set -euo pipefail

GIT_ROOT="$(git rev-parse --show-toplevel)"
cd "$GIT_ROOT"

TOOLS_PATH="$GIT_ROOT/.tools"
RELEASE="master-1d7ec53d"
CLANG_FORMAT_VERSION="15"
CLANG_FORMAT_PATH="$TOOLS_PATH/clang-format-$CLANG_FORMAT_VERSION-$RELEASE"

mkdir -p "$TOOLS_PATH"

error() {
  >&2 echo "$@"
  exit 1
}

get_os() {
  UNAME_OUTPUT="$(uname -s)"
  case "$UNAME_OUTPUT" in
    Linux*)
      OS=Linux
      ;;
    Darwin*)
      OS=Mac
      ;;
    *)
      error "Unknown OS $UNAME_OUTPUT. Exiting..."
  esac

  echo "$OS"
}

download_clang_tool() {
  TOOL="$1"
  VERSION="$2"
  TARGET_PATH="$3"

  BASE_URL="https://github.com/muttleyxd/clang-tools-static-binaries/releases/download/$RELEASE/"

  OS="$(get_os)"
  case "$OS" in
    Linux)
      URL_OS="linux"
      ;;
    Mac)
      URL_OS="macosx"
      ;;
    *)
      error "Unknown return value from get_os: $OS. Exiting..."
  esac
  URL="$BASE_URL/clang-${TOOL}-${VERSION}_${URL_OS}-amd64"
  echo "Downloading from $URL..."

  if command -v wget &> /dev/null; then
    wget "$URL" -O "$TARGET_PATH"
  elif command -v curl &> /dev/null; then
    curl -L "$URL" -o "$TARGET_PATH"
  else
    error "Could not find either wget or curl. Exiting..."
  fi
}

if [[ ! -e $CLANG_FORMAT_PATH ]]; then
  download_clang_tool format "$CLANG_FORMAT_VERSION" "$CLANG_FORMAT_PATH"
  chmod u+x "$CLANG_FORMAT_PATH"
fi

mapfile -t FILES < <(git ls-files ':!:triton/**' '*.h' '*.cc' '*.cpp' '*.cu' '*.c')
"$CLANG_FORMAT_PATH" -i "${FILES[@]}"
