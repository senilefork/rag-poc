#!/bin/sh
# Sync Python dependencies from requirements.txt at container start.
#
# docker-compose bind-mounts the repo over /app, so edits to .py files are already
# live without a rebuild. requirements.txt is the exception: it is baked into the
# image at build time. Installing it here means editing it no longer requires
# `docker compose build`, which otherwise re-runs `docling-tools models download`.
#
# Skipped when requirements.txt is byte-identical to the last install, tracked by
# its hash in the deps_cache volume. PIP_CACHE_DIR lives in that volume too, so a
# real dependency change downloads only the new wheels.
set -e

MARKER=/opt/deps/requirements.sha256
REQUIREMENTS=/app/requirements.txt

if [ -f "$REQUIREMENTS" ]; then
	current=$(sha256sum "$REQUIREMENTS" | cut -d' ' -f1)
	cached=$(cat "$MARKER" 2>/dev/null || true)

	if [ "$current" != "$cached" ]; then
		echo "entrypoint: requirements.txt changed -> installing dependencies"
		pip install --quiet --disable-pip-version-check --root-user-action=ignore \
			-r "$REQUIREMENTS"
		printf '%s' "$current" >"$MARKER"
	else
		echo "entrypoint: requirements.txt unchanged -> skipping install"
	fi
else
	echo "entrypoint: no $REQUIREMENTS -> skipping install"
fi

exec "$@"