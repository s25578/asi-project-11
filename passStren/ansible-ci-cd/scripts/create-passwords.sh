#!/usr/bin/env bash

while read -rp 'Username: ' username && read -rsp 'Password: ' password; do
	docker run --rm --entrypoint htpasswd httpd:2 \
		-Bbn "$username" "$password"
	echo >&2
done
